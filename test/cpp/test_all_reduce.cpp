#include "all_reduce_fusion_impl.h"

namespace test {

namespace comm_utils {

class Communicator {
public:
    Communicator() :
        initialized_(false) {
    }

    void local_init(
        int64_t device_id,     // assign device
        int64_t rank,          // rank in group
        int64_t world_size,    // group size
        int64_t size_in_bytes, // private data size
        int64_t max_thread_blocks = NBLOCKS_PER_GPU) {
        assert(rank < world_size);
        device_id_ = device_id;
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        max_thread_blocks_ = max_thread_blocks;
        gpuSetDevice(device_id_);
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuDeviceSynchronize();
        initialized_ = true;
    }

    friend void init_communicators(std::vector<Communicator> &communicators);

    int get_dst_id(int i) const {
        return (rank_ + i) % world_size_;
    }

    std::tuple<CommMeta, CommPtrs> get_comm_data() const {
        assert(initialized_);
        CommMeta meta;
        for (int r = 0; r < world_size_; ++r) {
            meta.barrier_flag_ptrs[r] = comm_barrier_flags_[r];
        }
        meta.sync_clock = sync_clock_;
        meta.rank = rank_;
        meta.nranks = world_size_;
        CommPtrs cptrs;
        for (int r = 0; r < world_size_; ++r) {
            cptrs.data_ptrs[r] = comm_data_[r];
        }
        return {meta, cptrs};
    }

    ~Communicator() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
    }

private:
    int device_id_;
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int max_thread_blocks_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> comm_barrier_flags_;
    std::vector<void *> comm_data_;
    bool initialized_;
};

void init_communicators(std::vector<Communicator> &communicators) {
    int world_size = communicators[0].world_size_;
    std::vector<void *> comm_barrier_flags, comm_data;
    comm_barrier_flags.resize(world_size);
    comm_data.resize(world_size);
    for (int i = 0; i < communicators.size(); ++i) {
        assert(communicators[i].initialized_);
        assert(communicators[i].world_size_ == world_size);
        communicators[i].comm_barrier_flags_.resize(world_size);
        communicators[i].comm_data_.resize(world_size);
        comm_barrier_flags[i] = communicators[i].barrier_flags_;
        comm_data[i] = communicators[i].data_;
    }
    for (int i = 0; i < communicators.size(); ++i) {
        for (int r = 0; r < world_size; ++r) {
            communicators[i].comm_barrier_flags_[r] = comm_barrier_flags[communicators[i].get_dst_id(r)];
            communicators[i].comm_data_[r] = comm_data[communicators[i].get_dst_id(r)];
        }
    }
}

template <typename T, int NRanks, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_direct_kernel(int size, CommDeviceMeta<NRanks> meta, CommPtrs cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    SyncComm<NRanks> comm(meta);
    using vec_t_ = kernel_utils::vec_t<T, VEC_SIZE>;
    using acc_vec_t_ = kernel_utils::vec_t<float, VEC_SIZE>;
    comm.template sync<true, false>();
    vec_t_ vec;
    acc_vec_t_ acc_vec;
    for (
        int idx = ((meta.rank + blockIdx.x * NRanks) * BLOCK_SIZE + threadIdx.x) * VEC_SIZE;
        idx < size;
        idx += gridDim.x * NRanks * BLOCK_SIZE * VEC_SIZE) {
        vec.load(reinterpret_cast<T *>(cptrs.data_ptrs[0]) + idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            acc_vec[v] = (float)vec[v];
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
            vec.load(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + idx);
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc_vec[v] += (float)vec[v];
            }
        }
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            vec[v] = (T)acc_vec[v];
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vec.store(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + idx);
        }
    }
    comm.template sync<true, true>();
}

template <typename T, int NRanks>
void allreduce_kernel_launcher(T *allreduce_in, T *allreduce_out, int size,
                               CommDeviceMeta<NRanks> &meta, CommPtrs &cptrs,
                               gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCK_WORK_SIZE = NRanks * BLOCK_SIZE * VEC_SIZE;
    int nblocks = (size + BLOCK_WORK_SIZE - 1) / BLOCK_WORK_SIZE;
    nblocks = std::min(nblocks, NBLOCKS_PER_GPU);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(nblocks);
    gpuMemcpyAsync(cptrs.data_ptrs[0], allreduce_in, size * sizeof(T), gpuMemcpyDeviceToDevice, stream);
    allreduce_direct_kernel<T, NRanks, BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(size, meta, cptrs);
    gpuMemcpyAsync(allreduce_out, cptrs.data_ptrs[0], size * sizeof(T), gpuMemcpyDeviceToDevice, stream);
}

template <typename T>
void allreduce_kernel(T *allreduce_in, T *allreduce_out, int size,
                      CommMeta &meta, CommPtrs &cptrs,
                      gpuStream_t stream = 0) {
#define DISPATCH_NRANKS(NRANKS)                                                                        \
    {                                                                                                  \
        CommDeviceMeta<NRANKS> dmeta;                                                                  \
        for (int i = 0; i < NRANKS; ++i) {                                                             \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                                    \
        }                                                                                              \
        dmeta.sync_clock = meta.sync_clock;                                                            \
        dmeta.rank = meta.rank;                                                                        \
        dmeta.nranks = meta.nranks;                                                                    \
        allreduce_kernel_launcher<T, NRANKS>(allreduce_in, allreduce_out, size, dmeta, cptrs, stream); \
    }
    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        assert(false);
    }
#undef DISPATCH_NRANKS
}

} // namespace comm_utils

template <typename T>
class CPUInputs {
public:
    int rank;
    int size;
    T *allreduce_in;
    T *allreduce_out;

    CPUInputs() :
        size(0), allreduce_in(nullptr), allreduce_out(nullptr) {
    }

    void allocate() {
        allreduce_in = new T[size];
        allreduce_out = new T[size];
    }

    void reset() {
        for (int i = 0; i < size; ++i) {
            allreduce_in[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
    }

    ~CPUInputs() {
        delete[] allreduce_in;
        delete[] allreduce_out;
    }
};

template <typename T>
class GPUInputs {
public:
    int rank;
    int size;
    T *allreduce_in;
    T *allreduce_out;

    GPUInputs() :
        size(0), allreduce_in(nullptr), allreduce_out(nullptr) {
    }

    void allocate() {
        gpuMalloc(&allreduce_in, size * sizeof(T));
        gpuMalloc(&allreduce_out, size * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T> &inputs) {
        gpuMemcpy(allreduce_in, inputs.allreduce_in, size * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(allreduce_in);
        gpuFree(allreduce_out);
        gpuDeviceSynchronize();
    }

    bool is_error(T out, T ref, float atol) {
        return std::isnan((float)out) || std::abs((float)out - (float)ref) > atol;
    }

    bool validate(CPUInputs<T> &inputs, float atol) {
        auto allreduce_out_cpu = new T[size];
        gpuMemcpy(allreduce_out_cpu, allreduce_out, size * sizeof(T), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
        bool val = true;
        for (int i = 0; i < size; ++i) {
            if (is_error(allreduce_out_cpu[i], inputs.allreduce_out[i], atol)) {
                val = false;
                std::cout << "\n>>> allreduce_out:" << (float)allreduce_out_cpu[i] << ", allreduce_out_ref:" << (float)inputs.allreduce_out[i] << "\n";
                break;
            }
        }
        delete[] allreduce_out_cpu;
        return val;
    }
};

template <typename T>
void allreduce_ref(std::vector<CPUInputs<T>> &inputs) {
    int size = inputs[0].size;
    int nranks = inputs.size();
    auto allreduce_out = new float[size];
    // get rank 0
    for (int i = 0; i < size; ++i) {
        allreduce_out[i] = (float)inputs[0].allreduce_in[i];
    }
    // reduce all ranks
    for (int r = 1; r < nranks; ++r) {
        for (int i = 0; i < size; ++i) {
            allreduce_out[i] += (float)inputs[r].allreduce_in[i];
        }
    }
    // scatter
    for (int r = 0; r < nranks; ++r) {
        for (int i = 0; i < size; ++i) {
            inputs[r].allreduce_out[i] = (T)allreduce_out[i];
        }
    }
    delete[] allreduce_out;
}

template <typename T>
std::tuple<float, float> allreduce_device(std::vector<GPUInputs<T>> &inputs, std::vector<comm_utils::Communicator> &communicators) {
    int nranks = inputs.size();
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuDeviceSynchronize();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        auto comm_data = communicators[r].get_comm_data();
        comm_utils::allreduce_kernel<T>(
            inputs[r].allreduce_in,
            inputs[r].allreduce_out,
            inputs[r].size,
            std::get<0>(comm_data),
            std::get<1>(comm_data));
    }
    for (int r = 0; r < nranks; ++r) {
        gpuSetDevice(r);
        gpuDeviceSynchronize();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    size_t nbytes_total = (nranks - 1) * 2 * inputs[0].size * sizeof(T);
    double gbps = ((double)nbytes_total / seconds) / 1e9;
    return {seconds, gbps};
}

template <typename T>
std::tuple<bool, float, float> runbench(int nranks, int size, bool validate, float atol = 1e-3) {
    std::vector<comm_utils::Communicator> communicators;
    std::vector<CPUInputs<T>> cpu_inputs;
    std::vector<GPUInputs<T>> gpu_inputs;
    communicators.resize(nranks);
    cpu_inputs.resize(nranks);
    gpu_inputs.resize(nranks);
    for (int r = 0; r < nranks; ++r) {
        communicators[r].local_init(r, r, nranks, size * sizeof(T));
        cpu_inputs[r].rank = r;
        cpu_inputs[r].size = size;
        cpu_inputs[r].allocate();
        cpu_inputs[r].reset();
        gpu_inputs[r].rank = r;
        gpu_inputs[r].size = size;
        gpu_inputs[r].allocate();
        gpu_inputs[r].reset(cpu_inputs[r]);
    }
    init_communicators(communicators);
    auto [dur, gbps] = allreduce_device<T>(gpu_inputs, communicators);
    bool val = true;
    if (validate) {
        allreduce_ref<T>(cpu_inputs);
        for (int r = 0; r < nranks; ++r) {
            gpuSetDevice(r);
            val = val && gpu_inputs[r].validate(cpu_inputs[r], atol);
        }
    }
    return {val, dur, gbps};
}

} // namespace test

int main() {
    int nranks = comm_utils::enable_p2p();
    std::cout << "nranks:" << nranks << "\n";
    std::vector<int> warmup_sizes = {1024 * 1024, 1024 * 1024};
    for (auto size : warmup_sizes) {
        auto [val, dur, gbps] = test::runbench<float>(nranks, size, false, 1e-2);
    }
    std::cout << "====================================\n";

    {
        std::vector<int> sizes = {
            14500 * 4096,
            14500 * 4096,
        };
        using T = __bfloat16;
        for (auto size : sizes) {
            auto [val, dur, gbps] = test::runbench<T>(nranks, size, true, 1e-2);
            std::cout << "size:" << size << ", dtype:" << typeid(T).name();
            std::cout << ", val:" << val << ", dur_s:" << dur << ", gbps:" << gbps << "\n";
        }
    }
}
