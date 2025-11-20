#include "all_reduce_fusion_impl.h"

namespace test {

template <typename T>
class GPUInputs {
public:
    int size;
    int hidden_dim;
    int rank;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;

    GPUInputs() :
        size(0), hidden_dim(0), allreduce_in(nullptr), residual_in(nullptr),
        residual_out(nullptr), norm_out(nullptr), rms_gamma(nullptr) {
    }

    void allocate(int rank, int size, int hidden_dim) {
        this->size = size;
        this->hidden_dim = hidden_dim;
        this->rank = rank;
        gpuSetDevice(rank);
        gpuMalloc(&allreduce_in, size * sizeof(T));
        gpuMalloc(&residual_in, size * sizeof(T));
        gpuMalloc(&residual_out, size * sizeof(T));
        gpuMalloc(&norm_out, size * sizeof(T));
        gpuMalloc(&rms_gamma, hidden_dim * sizeof(T));
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuSetDevice(rank);
        gpuFree(allreduce_in);
        gpuFree(residual_in);
        gpuFree(residual_out);
        gpuFree(norm_out);
        gpuFree(rms_gamma);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUCommWorkspace {
    int nblocks;
    int rank;
    int nranks;
    int size;

public:
    int *counter;
    void *twoshot_comm_bufs;
    int *twoshot_barrier_flags;
    int *twoshot_sync_clock;
    // oneshot
    void *oneshot_comm_bufs;
    int *oneshot_sync_clock;
    int *oneshot_clear;
    int *oneshot_comm_size;

    GPUCommWorkspace() :
        nblocks(NBLOCKS_PER_GPU), counter(nullptr), twoshot_comm_bufs(nullptr),
        twoshot_barrier_flags(nullptr), twoshot_sync_clock(nullptr),
        oneshot_comm_bufs(nullptr), oneshot_sync_clock(nullptr),
        oneshot_clear(nullptr), oneshot_comm_size(nullptr) {
    }

    void allocate(int rank, int nranks, int size) {
        this->rank = rank;
        this->nranks = nranks;
        this->size = size;
        gpuSetDevice(rank);
        gpuMalloc(&counter, sizeof(int));
        gpuMalloc(&twoshot_comm_bufs, 2 * size * sizeof(T));
        gpuMalloc(&twoshot_barrier_flags, nblocks * nranks * sizeof(int));
        gpuMalloc(&twoshot_sync_clock, sizeof(int));
        // oneshot
        gpuMalloc(&oneshot_comm_bufs, 3 * nranks * size * sizeof(T));
        gpuMalloc(&oneshot_sync_clock, sizeof(int));
        gpuMalloc(&oneshot_clear, sizeof(int));
        gpuMalloc(&oneshot_comm_size, sizeof(int));
        gpuDeviceSynchronize();
        reset();
    }

    void reset() {
        gpuSetDevice(rank);
        gpuMemset(counter, 0, sizeof(int));
        gpuMemset(twoshot_barrier_flags, 0, nblocks * nranks * sizeof(int));
        gpuMemset(twoshot_sync_clock, 0, sizeof(int));
        // oneshot
        gpuMemset(oneshot_sync_clock, 0, sizeof(int));
        int clear_size = nranks * size;
        int comm_size = nranks * size * (int)sizeof(T); // large size
        // gpuMemcpy(oneshot_clear, &clear_size, sizeof(int),
        // gpuMemcpyHostToDevice);
        gpuMemset(oneshot_clear, 0, sizeof(int));
        gpuMemcpy(oneshot_comm_size, &comm_size, sizeof(int),
                  gpuMemcpyHostToDevice);
        T *lamport_data_bufs_ = new T[3 * nranks * size];
        for (int i = 0; i < 3 * nranks * size; ++i) {
            lamport_data_bufs_[i] = neg_zero_v<T>;
        }
        gpuMemcpy(oneshot_comm_bufs, lamport_data_bufs_,
                  3 * nranks * size * sizeof(T), gpuMemcpyHostToDevice);
        delete[] lamport_data_bufs_;
        gpuDeviceSynchronize();
    }

    ~GPUCommWorkspace() {
        gpuSetDevice(rank);
        gpuFree(counter);
        gpuFree(twoshot_comm_bufs);
        gpuFree(twoshot_barrier_flags);
        gpuFree(twoshot_sync_clock);
        // oneshot
        gpuFree(oneshot_comm_bufs);
        gpuFree(oneshot_sync_clock);
        gpuFree(oneshot_clear);
        gpuFree(oneshot_comm_size);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUWorkSpace {
public:
    GPUWorkSpace() :
        workspace_(nullptr) {
    }

    void init(std::vector<GPUCommWorkspace<T>> &rs, int rank) {
        gpuSetDevice(rank);
        rank_ = rank;
        int nranks = rs.size();
        auto &r = rs[rank];
        std::vector<void *> workspace(nranks * 3 + 5);
        for (int peer = 0; peer < nranks; ++peer) {
            workspace[peer] = (void *)rs[peer].twoshot_comm_bufs;
            workspace[nranks + peer] = (void *)rs[peer].twoshot_barrier_flags;
            // oneshot
            workspace[2 * nranks + peer] = (void *)rs[peer].oneshot_comm_bufs;
        }
        workspace[nranks * 3 + 0] = (void *)r.counter;
        workspace[nranks * 3 + 1] = (void *)r.twoshot_sync_clock;
        workspace[nranks * 3 + 2] = (void *)r.oneshot_sync_clock;
        workspace[nranks * 3 + 3] = (void *)r.oneshot_comm_size;
        workspace[nranks * 3 + 4] = (void *)r.oneshot_clear;
        gpuMalloc(&workspace_, workspace.size() * sizeof(void *));
        gpuMemcpy(workspace_, workspace.data(), workspace.size() * sizeof(void *),
                  gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUWorkSpace() {
        gpuSetDevice(rank_);
        gpuFree(workspace_);
        gpuDeviceSynchronize();
    }

    void **workspace() const {
        return workspace_;
    }

private:
    void **workspace_;
    int rank_;
};

void allreduce_rmsnorm_ref(const float *allreduce_in, const float *residual_in,
                           const float *rms_gamma, int size, int hidden_dim,
                           int nranks, float *residual_out, float *norm_out,
                           float eps = 1e-6) {
    auto allreduce_out = new float[size];
    // get rank 0
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] = allreduce_in[i];
    }
    // reduce all ranks
    for (int r = 1; r < nranks; ++r) {
        for (int i = 0; i < size; ++i) {
            allreduce_out[i] += allreduce_in[r * size + i];
        }
    }
    // residual
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] += residual_in[i];
        residual_out[i] = allreduce_out[i];
    }
    // norm
    int num_tokens = size / hidden_dim;
    for (int t = 0; t < num_tokens; ++t) {
        double x2 = 0;
        int offset_token = t * hidden_dim;
        for (int h = 0; h < hidden_dim; ++h) {
            auto data = allreduce_out[offset_token + h];
            x2 += data * data;
        }
        double beta = (double)1.0 / std::sqrt(x2 / hidden_dim + eps);
        for (int h = 0; h < hidden_dim; ++h) {
            norm_out[offset_token + h] = allreduce_out[offset_token + h] * beta;
            norm_out[offset_token + h] *= rms_gamma[h];
        }
    }
    delete[] allreduce_out;
}

void runbench(int nranks, int size, int hidden_dim, float eps = 1e-6,
              float atol = 0.0001) {
    // input
    auto allreduce_in = new float[nranks * size];
    auto residual_in = new float[size];
    auto rms_gamma = new float[hidden_dim];

    // output
    auto residual_out_ref = new float[size];
    auto norm_out_ref = new float[size];
    auto residual_out = new float[size];
    auto norm_out = new float[size];

    // gen data
    for (int i = 0; i < nranks * size; ++i) {
        allreduce_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < size; ++i) {
        residual_in[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }
    for (int i = 0; i < hidden_dim; ++i) {
        rms_gamma[i] = 0.f + 1.f * (rand() / (float)INT_MAX);
    }

    std::vector<GPUInputs<float>> gpu_inputs(nranks);
    std::vector<GPUCommWorkspace<float>> comm_workspaces(nranks);
    for (int r = 0; r < nranks; ++r) {
        gpu_inputs[r].allocate(r, size, hidden_dim);
        comm_workspaces[r].allocate(r, nranks, size);
    }

    // gen gpu data
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(gpu_inputs[r].allreduce_in, allreduce_in + r * size,
                  size * sizeof(float), gpuMemcpyHostToDevice);
        gpuMemcpy(gpu_inputs[r].residual_in, residual_in, size * sizeof(float),
                  gpuMemcpyHostToDevice);
        gpuMemcpy(gpu_inputs[r].rms_gamma, rms_gamma, hidden_dim * sizeof(float),
                  gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    std::vector<GPUWorkSpace<float>> workspaces(nranks);
    for (int r = 0; r < nranks; ++r) {
        workspaces[r].init(comm_workspaces, r);
    }

    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        allreduce_rms_fusion_impl<float>(
            workspaces[rank].workspace(), rank, nranks, size, hidden_dim,
            gpu_inputs[rank].allreduce_in, gpu_inputs[rank].residual_in,
            gpu_inputs[rank].residual_out, gpu_inputs[rank].norm_out,
            gpu_inputs[rank].rms_gamma, eps);
    }
    for (int rank = 0; rank < nranks; ++rank) {
        gpuSetDevice(rank);
        gpuDeviceSynchronize();
    }

    allreduce_rmsnorm_ref(allreduce_in, residual_in, rms_gamma, size, hidden_dim,
                          nranks, residual_out_ref, norm_out_ref, eps);

    bool val = true;
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuMemcpy(residual_out, gpu_inputs[r].residual_out, size * sizeof(float),
                  gpuMemcpyDeviceToHost);
        gpuMemcpy(norm_out, gpu_inputs[r].norm_out, size * sizeof(float),
                  gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        for (int i = 0; i < size; ++i) {
            if (std::isnan(residual_out[i]) || std::abs(residual_out[i] - residual_out_ref[i]) > atol) {
                std::cout << "residual_out:" << residual_out[i]
                          << ", residual_out_ref:" << residual_out_ref[i] << "\n";
                val = false;
                break;
            }
            if (std::isnan(norm_out[i]) || std::abs(norm_out[i] - norm_out_ref[i]) > atol) {
                std::cout << "norm_out:" << norm_out[i]
                          << ", norm_out_ref:" << norm_out_ref[i] << "\n";
                val = false;
                break;
            }
        }
    }
    std::cout << "validation:" << val << "\n";

    delete[] allreduce_in;
    delete[] residual_in;
    delete[] rms_gamma;
    delete[] residual_out_ref;
    delete[] norm_out_ref;
    delete[] residual_out;
    delete[] norm_out;
}

} // namespace test

int main() {
    int nranks = enable_p2p();
    std::cout << "nranks:" << nranks << "\n";
    std::vector<int> num_tokens_ = {513, 1257, 127, 778, 10024, 3};
    std::vector<int> hidden_dims = {1024, 512 - 4, 124};
    for (auto num_tokens : num_tokens_) {
        for (auto hidden_dim : hidden_dims) {
            int size = num_tokens * hidden_dim;
            std::cout << "num_tokens:" << num_tokens << ", hidden_dim:" << hidden_dim
                      << "\n";
            test::runbench(nranks, size, hidden_dim);
        }
    }
}
