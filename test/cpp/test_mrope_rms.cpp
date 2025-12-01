#include "rope_rms_impl.h"

namespace test {

template <typename T, int M>
class CPUInputs {
    int64_t num_heads_;

public:
    int64_t num_tokens;
    int64_t num_heads_q;
    int64_t num_heads_k;
    int64_t num_heads_v;
    int64_t head_size;
    bool is_neox_style;
    double eps;
    std::array<int64_t, M> mrope_section;
    int64_t max_positions;
    bool is_interleaved;
    T *qkv;
    T *q_w;
    T *k_w;
    T *cos_sin;         // max_positions, head_size
    int64_t *positions; // M, num_tokens

    CPUInputs(
        int64_t num_tokens,
        int64_t num_heads_q,
        int64_t num_heads_k,
        int64_t num_heads_v,
        int64_t head_size,
        bool is_neox_style,
        double eps,
        std::array<int64_t, M> mrope_section,
        int64_t max_positions,
        bool is_interleaved) :
        num_tokens(num_tokens),
        num_heads_q(num_heads_q), num_heads_k(num_heads_k), num_heads_v(num_heads_v),
        head_size(head_size), is_neox_style(is_neox_style), eps(eps), mrope_section(mrope_section),
        max_positions(max_positions), is_interleaved(is_interleaved) {
        num_heads_ = num_heads_q + num_heads_k + num_heads_v;
        auto dim = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
        assert(dim == head_size / 2);
    }

    void allocate() {
        qkv = new T[num_tokens * num_heads_ * head_size];
        q_w = new T[head_size];
        k_w = new T[head_size];
        cos_sin = new T[max_positions * head_size];
        positions = new int64_t[M * num_tokens];
    }

    void reset() {
        for (int i = 0; i < num_tokens * num_heads_ * head_size; ++i) {
            qkv[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < head_size; ++i) {
            q_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            k_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < max_positions * head_size; ++i) {
            cos_sin[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < M * num_tokens; ++i) {
            positions[i] = rand() % max_positions;
        }
    }

    ~CPUInputs() {
        delete[] qkv;
        delete[] q_w;
        delete[] k_w;
        delete[] cos_sin;
        delete[] positions;
    }

    void process_token(T *data, T *weight, T *cos_sin, int64_t num_head, int64_t head_size, double eps, bool is_neox_style) {
        auto half_head_size = head_size / 2;
        for (auto hid = 0; hid < num_head; ++hid) {
            double x2 = 0;
            auto offset = hid * head_size;
            for (int h = 0; h < head_size; ++h) {
                auto x = data[offset + h];
                x2 += x * x;
            }
            double beta = (double)1.0 / std::sqrt(x2 / head_size + eps);
            for (int h = 0; h < half_head_size; ++h) {
                auto cos = cos_sin[h];
                auto sin = cos_sin[h + half_head_size];
                if (is_neox_style) {
                    auto x0 = data[offset + h] * beta * weight[h];
                    auto x1 = data[offset + h + half_head_size] * beta * weight[h + half_head_size];
                    data[offset + h] = x0 * cos - x1 * sin;
                    data[offset + h + half_head_size] = x1 * cos + x0 * sin;
                } else {
                    auto x0 = data[offset + h * 2 + 0] * beta * weight[h * 2 + 0];
                    auto x1 = data[offset + h * 2 + 1] * beta * weight[h * 2 + 1];
                    data[offset + h * 2 + 0] = x0 * cos - x1 * sin;
                    data[offset + h * 2 + 1] = x1 * cos + x0 * sin;
                }
            }
        }
    }

    void operator()() {
        for (auto tid = 0; tid < num_tokens; ++tid) {
            auto cos_sin_ = new T[head_size];
            auto half_head_size = head_size / 2;
            if (is_interleaved) {
                // cos
                for (int i = 0; i < half_head_size; ++i) {
                    auto im = i % M;
                    if (im >= 1 && i < mrope_section[im] * M) {
                        auto p = positions[im * num_tokens + tid];
                        cos_sin_[i] = cos_sin[p * head_size + i];
                    } else {
                        cos_sin_[i] = cos_sin[positions[tid] * head_size + i];
                    }
                }
                // sin
                for (int i = 0; i < half_head_size; ++i) {
                    auto im = i % M;
                    if (im >= 1 && i < mrope_section[im] * M) {
                        auto p = positions[im * num_tokens + tid];
                        cos_sin_[half_head_size + i] = cos_sin[p * head_size + half_head_size + i];
                    } else {
                        cos_sin_[half_head_size + i] = cos_sin[positions[tid] * head_size + half_head_size + i];
                    }
                }
            } else {
                int i = 0;
                int end = 0;
                // cos
                for (int mid = 0; mid < M; ++mid) {
                    end += mrope_section[mid];
                    auto p = positions[mid * num_tokens + tid];
                    for (; i < end; ++i) {
                        cos_sin_[i] = cos_sin[p * head_size + i];
                    }
                }
                // sin
                for (int mid = 0; mid < M; ++mid) {
                    end += mrope_section[mid];
                    auto p = positions[mid * num_tokens + tid];
                    for (; i < end; ++i) {
                        cos_sin_[i] = cos_sin[p * head_size + i];
                    }
                }
            }
            auto q = &qkv[tid * num_heads_ * head_size];
            auto k = &q[num_heads_q * head_size];
            process_token(q, q_w, cos_sin_, this->num_heads_q, this->head_size, this->eps, this->is_neox_style);
            process_token(k, k_w, cos_sin_, this->num_heads_k, this->head_size, this->eps, this->is_neox_style);
            delete[] cos_sin_;
        }
    }
};

template <typename T, int M>
class GPUInputs {
    int64_t num_heads_;

public:
    int64_t num_tokens;
    int64_t num_heads_q;
    int64_t num_heads_k;
    int64_t num_heads_v;
    int64_t head_size;
    bool is_neox_style;
    double eps;
    std::array<int64_t, M> mrope_section;
    int64_t max_positions;
    bool is_interleaved;
    T *qkv;
    T *q_w;
    T *k_w;
    T *cos_sin;         // max_positions, head_size
    int64_t *positions; // M, num_tokens

    GPUInputs(
        int64_t num_tokens,
        int64_t num_heads_q,
        int64_t num_heads_k,
        int64_t num_heads_v,
        int64_t head_size,
        bool is_neox_style,
        double eps,
        std::array<int64_t, M> mrope_section,
        int64_t max_positions,
        bool is_interleaved) :
        num_tokens(num_tokens),
        num_heads_q(num_heads_q), num_heads_k(num_heads_k), num_heads_v(num_heads_v),
        head_size(head_size), is_neox_style(is_neox_style), eps(eps), mrope_section(mrope_section),
        max_positions(max_positions), is_interleaved(is_interleaved) {
        num_heads_ = num_heads_q + num_heads_k + num_heads_v;
        auto dim = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
        assert(dim == head_size / 2);
    }

    void allocate() {
        gpuMalloc(&qkv, num_tokens * num_heads_ * head_size * sizeof(T));
        gpuMalloc(&q_w, head_size * sizeof(T));
        gpuMalloc(&k_w, head_size * sizeof(T));
        gpuMalloc(&cos_sin, max_positions * head_size * sizeof(T));
        gpuMalloc(&positions, M * num_tokens * sizeof(int64_t));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T, M> &inputs) {
        gpuMemcpy(qkv, inputs.qkv, num_tokens * num_heads_ * head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(q_w, inputs.q_w, head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(k_w, inputs.k_w, head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(cos_sin, inputs.cos_sin, max_positions * head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(positions, inputs.positions, M * num_tokens * sizeof(int64_t), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(qkv);
        gpuFree(q_w);
        gpuFree(k_w);
        gpuFree(cos_sin);
        gpuFree(positions);
        gpuDeviceSynchronize();
    }

    float operator()() {
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        gpuEventRecord(start);

        rope_rms::fused_mrope_rms<T, M>(
            qkv, q_w, k_w, cos_sin, positions, num_tokens, 1, num_tokens, num_heads_q, num_heads_k, num_heads_v,
            head_size, is_neox_style, eps, mrope_section, is_interleaved, 0);
        gpuDeviceSynchronize();

        gpuEventRecord(stop);
        gpuEventSynchronize(stop);
        float ms = 0;
        gpuEventElapsedTime(&ms, start, stop);
        float input_bytes = num_tokens * (num_heads_q + num_heads_k) * head_size * sizeof(T)
                            + 2 * head_size * sizeof(T)
                            + num_tokens * head_size * sizeof(T);
        float output_bytes = num_tokens * (num_heads_q + num_heads_k) * head_size * sizeof(T);
        float gbps = (input_bytes + output_bytes) / 1000.0 / 1000.0 / ms;
        return gbps;
    }

    bool is_error(T out, T ref, float atol) {
        return std::isnan(out) || std::abs(out - ref) > atol;
    }

    bool validate(CPUInputs<T, M> &inputs, float atol) {
        auto out_qkv_cpu = new T[num_tokens * num_heads_ * head_size];
        gpuMemcpy(out_qkv_cpu, qkv, num_tokens * num_heads_ * head_size * sizeof(T), gpuMemcpyDeviceToHost);
        bool val = true;
        for (int i = 0; i < num_tokens * num_heads_ * head_size; ++i) {
            if (is_error(out_qkv_cpu[i], inputs.qkv[i], atol)) {
                val = false;
                std::cout << "\n>>> out_qkv:" << out_qkv_cpu[i] << ", ref_qkv:" << inputs.qkv[i] << "\n";
                break;
            }
        }
        delete[] out_qkv_cpu;
        return val;
    }
};

template <typename T, int M>
std::tuple<bool, float> runbench(
    int64_t num_tokens,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_size,
    bool is_neox_style,
    double eps,
    std::array<int64_t, M> mrope_section,
    int64_t max_positions,
    bool is_interleaved,
    float atol = 0.0001) {
    CPUInputs<T, M> cpu_inputs(num_tokens, num_heads_q, num_heads_k, num_heads_v, head_size, is_neox_style, eps, mrope_section, max_positions, is_interleaved);
    GPUInputs<T, M> gpu_inputs(num_tokens, num_heads_q, num_heads_k, num_heads_v, head_size, is_neox_style, eps, mrope_section, max_positions, is_interleaved);
    cpu_inputs.allocate();
    gpu_inputs.allocate();
    cpu_inputs.reset();
    gpu_inputs.reset(cpu_inputs);
    cpu_inputs();
    float gbps = gpu_inputs();
    bool val = gpu_inputs.validate(cpu_inputs, atol);
    return {val, gbps};
}

} // namespace test

int main() {
    constexpr int M = 3;
    std::vector<bool> is_neox_styles = {true, false};
    std::vector<int> num_tokens = {513, 1257, 127, 778, 10024, 3};
    std::vector<int> num_heads = {32, 64};
    std::vector<int> head_sizes = {64, 128, 256};
    std::vector<std::array<int64_t, M>> mrope_sections = {{12, 10, 10}, {24, 20, 20}, {48, 40, 40}};
    std::vector<bool> is_interleaveds = {true, false};
    double eps = 1e-6;
    int64_t max_positions = 10000;
    for (auto is_neox_style : is_neox_styles) {
        for (auto num_token : num_tokens) {
            for (auto num_head : num_heads) {
                int i = 0;
                for (auto head_size : head_sizes) {
                    auto ms = mrope_sections[i];
                    for (auto is_interleaved : is_interleaveds) {
                        std::cout << "num_token:" << num_token << ", num_head:" << num_head << ", head_size:" << head_size << ", is_neox_style:" << is_neox_style;
                        std::cout << ", is_interleaved:" << is_interleaved << ", mrope_section:(" << ms[0] << "," << ms[1] << "," << ms[2] << ")";
                        auto [val, gbps] = test::runbench<float, M>(num_token, num_head, num_head, num_head, head_size, is_neox_style, eps, ms, max_positions, is_interleaved);
                        std::cout << ", val:" << val << ", gbps:" << gbps << "\n";
                    }
                    ++i;
                }
            }
        }
    }
}
