#include "sgemm_impl.h"

namespace test {

template <typename T>
class CPUInputs {
public:
    int64_t m;
    int64_t n;
    int64_t k;
    double alpha;
    double beta;
    T *out;
    T *a;
    T *b;
    T *out_dev;
    T *a_dev;
    T *b_dev;

    CPUInputs(
        int64_t m,
        int64_t n,
        int64_t k,
        double alpha,
        double beta) :
        m(m),
        n(n), k(k), alpha(alpha), beta(beta) {
    }

    void allocate() {
        out = new T[m * n];
        a = new T[m * k];
        b = new T[k * n];
        gpuMalloc(&out_dev, m * n * sizeof(T));
        gpuMalloc(&a_dev, m * k * sizeof(T));
        gpuMalloc(&b_dev, k * n * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset() {
        for (int i = 0; i < m * n; ++i) {
            out[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < m * k; ++i) {
            a[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < k * n; ++i) {
            b[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        gpuMemcpy(out_dev, out, m * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(a_dev, a, m * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(b_dev, b, k * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~CPUInputs() {
        delete[] out;
        delete[] a;
        delete[] b;
        gpuFree(out_dev);
        gpuFree(a_dev);
        gpuFree(b_dev);
    }

    void operator()() {
        sgemm::sgemm_naive<T>(out_dev, a_dev, b_dev, m, n, k, alpha, beta, 0);
        gpuDeviceSynchronize();
        gpuMemcpy(out, out_dev, m * n * sizeof(T), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUInputs {
public:
    int64_t m;
    int64_t n;
    int64_t k;
    double alpha;
    double beta;
    T *out;
    T *a;
    T *b;

    GPUInputs(
        int64_t m,
        int64_t n,
        int64_t k,
        double alpha,
        double beta) :
        m(m),
        n(n), k(k), alpha(alpha), beta(beta) {
    }

    void allocate() {
        gpuMalloc(&out, m * n * sizeof(T));
        gpuMalloc(&a, m * k * sizeof(T));
        gpuMalloc(&b, k * n * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T> &inputs) {
        gpuMemcpy(out, inputs.out, m * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(a, inputs.a, m * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(b, inputs.b, k * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(out);
        gpuFree(a);
        gpuFree(b);
        gpuDeviceSynchronize();
    }

    std::tuple<float, float, float> operator()() {
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        gpuEventRecord(start);
        sgemm::sgemm<T>(out, a, b, m, n, k, alpha, beta, 0);
        gpuDeviceSynchronize();
        gpuEventRecord(stop);
        gpuEventSynchronize(stop);
        float ms = 0;
        gpuEventElapsedTime(&ms, start, stop);
        float input_bytes = (m * k + k * n + m * n) * sizeof(T);
        float output_bytes = (m * n) * sizeof(T);
        float gbps = (input_bytes + output_bytes) / 1000.0 / 1000.0 / ms;
        float tflops = ((float)2 * m * n * k) / (ms / 1000) * 1e-12;
        return {ms, gbps, tflops};
    }

    bool is_error(T out, T ref, float atol) {
        return std::isnan(out) || std::abs(out - ref) > atol;
    }

    bool validate(CPUInputs<T> &inputs, float atol) {
        auto out_cpu = new T[m * n];
        gpuMemcpy(out_cpu, out, m * n * sizeof(T), gpuMemcpyDeviceToHost);
        bool val = true;
        for (int i = 0; i < m * n; ++i) {
            if (is_error(out_cpu[i], inputs.out[i], atol)) {
                val = false;
                std::cout << "\n>>> out:" << out_cpu[i] << ", ref_out:" << inputs.out[i] << "\n";
                break;
            }
        }
        delete[] out_cpu;
        return val;
    }
};

template <typename T>
std::tuple<bool, float, float, float> runbench(
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    double beta,
    float atol = 0.0001) {
    CPUInputs<T> cpu_inputs(m, n, k, alpha, beta);
    GPUInputs<T> gpu_inputs(m, n, k, alpha, beta);
    cpu_inputs.allocate();
    gpu_inputs.allocate();
    cpu_inputs.reset();
    gpu_inputs.reset(cpu_inputs);
    cpu_inputs();
    auto r = gpu_inputs();
    bool val = gpu_inputs.validate(cpu_inputs, atol);
    return {val, std::get<0>(r), std::get<1>(r), std::get<2>(r)};
}

} // namespace test

int main() {
    std::vector<int> ms = {2048, 4096, 8192, 16384};
    std::vector<int> ns = {2048, 4096, 8192, 16384};
    std::vector<int> ks = {2048, 4096, 8192, 16384};
    double alpha = 1.0;
    double beta = 0.5;
    for (int i = 0; i < ms.size(); ++i) {
        auto m = ms[i];
        auto n = ns[i];
        auto k = ks[i];
        std::cout << "m:" << m << ", n:" << n << ", k:" << k << ", alpha:" << alpha << ", beta:" << beta;
        auto [val, ms, gbps, tflops] = test::runbench<float>(m, n, k, alpha, beta);
        std::cout << ", val:" << val << ", ms:" << ms << ", gbps:" << gbps << ", tflops:" << tflops << "\n";
    }
}
