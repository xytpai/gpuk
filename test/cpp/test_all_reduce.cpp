#include "all_reduce_fusion_impl.h"

namespace test {

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
