#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cassert>
#include <tuple>
#include <random>
#include <mma.h>
#include <cuda_fp16.h>
using namespace std;

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
    __device__ __forceinline__ scalar_t &operator[](int i) {
        return val[i];
    }
    __device__ __forceinline__ scalar_t const &operator[](int i) const {
        return val[i];
    }
};

template <typename scalar_t, typename acc_t>
struct MMA_M16N8K16 {
    using FragmentA = aligned_array<scalar_t, 8>;
    using FragmentB = aligned_array<scalar_t, 4>;
    using FragmentC = aligned_array<acc_t, 4>;

    __device__ __forceinline__ MMA_M16N8K16() {
        w_tid = threadIdx.x & 31;
    }

    __device__ __forceinline__ void operator()() {
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        float const *C = reinterpret_cast<float const *>(&c);
        float *D = reinterpret_cast<float *>(&c);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    }

    __device__ __forceinline__ void fill_fragment_c(acc_t val) {
        c.val[0] = val;
        c.val[1] = val;
        c.val[2] = val;
        c.val[3] = val;
    }

    __device__ __forceinline__ void load_matrix_a(scalar_t *ptr, int stride) {
#ifdef __CUDACC__
        auto A = reinterpret_cast<uint32_t *>(&a);
        auto addr = (uint32_t)__cvta_generic_to_shared(ptr + (w_tid % 16) * stride + (w_tid / 16) * 8);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
            : "r"(addr));
#else
        auto x = w_tid % 4 * 2;
        auto y = w_tid / 4;
        using vec_t = aligned_array<scalar_t, 2>;
        *reinterpret_cast<vec_t *>(&a.val[0]) = *reinterpret_cast<vec_t *>(&ptr[y * stride + x]);
        *reinterpret_cast<vec_t *>(&a.val[2]) = *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]);
        *reinterpret_cast<vec_t *>(&a.val[4]) = *reinterpret_cast<vec_t *>(&ptr[y * stride + 8 + x]);
        *reinterpret_cast<vec_t *>(&a.val[6]) = *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + 8 + x]);
#endif
    }

    __device__ __forceinline__ void load_matrix_b(scalar_t *ptr, int stride) {
#ifdef __CUDACC__
        auto B = reinterpret_cast<uint32_t *>(&b);
        auto addr = (uint32_t)__cvta_generic_to_shared(ptr + (w_tid % 16) * stride);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(B[0]), "=r"(B[1])
            : "r"(addr));
#else
        auto y = w_tid % 4 * 2;
        auto x = w_tid / 4;
        b.val[0] = ptr[y * stride + x];
        b.val[1] = ptr[(y + 1) * stride + x];
        b.val[2] = ptr[(8 + y) * stride + x];
        b.val[3] = ptr[(8 + y + 1) * stride + x];
#endif
    }

    __device__ __forceinline__ void store_matrix(acc_t *ptr, int stride, acc_t alpha, acc_t beta) {
        auto y = w_tid / 4;
        auto x = w_tid % 4 * 2;
        using vec_t = aligned_array<acc_t, 2>;
        auto vec0 = *reinterpret_cast<vec_t *>(&ptr[y * stride + x]);
        auto vec1 = *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]);
        vec0.val[0] = alpha * (acc_t)c.val[0] + beta * vec0.val[0];
        vec0.val[1] = alpha * (acc_t)c.val[1] + beta * vec0.val[1];
        vec1.val[0] = alpha * (acc_t)c.val[2] + beta * vec1.val[0];
        vec1.val[1] = alpha * (acc_t)c.val[3] + beta * vec1.val[1];
        *reinterpret_cast<vec_t *>(&ptr[y * stride + x]) = vec0;
        *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]) = vec1;
    }

private:
    int w_tid;
    FragmentA a;
    FragmentB b;
    FragmentC c;
};

template <typename scalar_t, int LOOP, int BLOCK_SIZE>
__global__ void wmma_loop_kernel(float *o, scalar_t *a, scalar_t *b) {
    constexpr int WM = 16;
    constexpr int WN = 8;
    constexpr int WK = 16;
    constexpr int BLOCK_WARPS = BLOCK_SIZE / 32;
    int wid = threadIdx.x / 32;
    int batch_id = blockIdx.x * BLOCK_WARPS + wid;
    auto o_ = o + batch_id * WM * WN;
    auto a_ = a + batch_id * WM * WK;
    auto b_ = b + batch_id * WK * WN;
    __shared__ scalar_t as[BLOCK_WARPS * WM * WK];
    __shared__ scalar_t bs[BLOCK_WARPS * WK * WN];
    auto as_ = as + wid * WM * WK;
    auto bs_ = bs + wid * WK * WN;
    if (threadIdx.x % 32 == 0) {
        for (int i = 0; i < WM * WK; ++i) {
            as_[i] = a_[i];
        }
        for (int i = 0; i < WK * WN; ++i) {
            bs_[i] = b_[i];
        }
    }
    __syncthreads();
    MMA_M16N8K16<scalar_t, float> mma;
    mma.fill_fragment_c(0);
    mma.load_matrix_a(as_, WK);
    mma.load_matrix_b(bs_, WN);
    for (int i = 0; i < LOOP; i++) {
        mma();
    }
    mma.store_matrix(o_, WN, 1, 0.5);
}

template <int LOOP, int NBLOCKS, int BLOCK_SIZE, bool VALID>
float wmma_test() {
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(NBLOCKS);
    constexpr int WM = 16;
    constexpr int WN = 8;
    constexpr int WK = 16;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    constexpr int BATCH_SIZE = NBLOCKS * WARPS_PER_BLOCK;
    constexpr int LEN_A = BATCH_SIZE * WM * WK;
    constexpr int LEN_B = BATCH_SIZE * WK * WN;
    constexpr int LEN_O = BATCH_SIZE * WM * WN;
    auto cpu_a = new __half[LEN_A];
    auto cpu_b = new __half[LEN_B];
    auto cpu_o = new float[LEN_O];
    auto ref_o = new float[LEN_O];
    for (int i = 0; i < LEN_A; ++i) {
        cpu_a[i] = (__half)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
    }
    for (int i = 0; i < LEN_B; ++i) {
        cpu_b[i] = (__half)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
    }
    for (int i = 0; i < LEN_O; ++i) {
        cpu_o[i] = (2.f * ((rand() / (float)INT_MAX) - 0.5f));
    }
    if constexpr (VALID) {
        assert(LOOP == 1);
        for (int b = 0; b < BATCH_SIZE; ++b) {
            auto a_ = cpu_a + b * WM * WK;
            auto b_ = cpu_b + b * WK * WN;
            auto o_ = cpu_o + b * WM * WN;
            auto d_ = ref_o + b * WM * WN;
            for (int m = 0; m < WM; ++m) {
                for (int n = 0; n < WN; ++n) {
                    float acc = 0.5 * o_[m * WN + n];
                    for (int k = 0; k < WK; ++k) {
                        acc += (float)a_[m * WK + k] * (float)b_[k * WN + n];
                    }
                    d_[m * WN + n] = acc;
                }
            }
        }
    }
    __half *gpu_a;
    __half *gpu_b;
    float *gpu_o;
    cudaMalloc(&gpu_a, LEN_A * sizeof(__half));
    cudaMalloc(&gpu_b, LEN_B * sizeof(__half));
    cudaMalloc(&gpu_o, LEN_O * sizeof(float));
    cudaMemcpy(gpu_a, cpu_a, LEN_A * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, LEN_B * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_o, cpu_o, LEN_O * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    wmma_loop_kernel<__half, LOOP, BLOCK_SIZE><<<numBlocks, threadsPerBlock>>>(gpu_o, gpu_a, gpu_b);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(cpu_o, gpu_o, LEN_O * sizeof(float), cudaMemcpyDeviceToHost);

    if constexpr (VALID) {
        float maxdiff = -1.0;
        for (int i = 0; i < LEN_O; ++i) {
            float diff = std::abs(ref_o[i] - cpu_o[i]);
            maxdiff = std::max(maxdiff, diff);
            // std::cout << "ref:" << ref_o[i] << ", out:" << cpu_o[i] << "\n";
        }
        std::cout << "maxdiff:" << maxdiff << "\n";
    }

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_o);
    delete[] cpu_a;
    delete[] cpu_b;
    delete[] cpu_o;
    delete[] ref_o;
    auto tflops = ((double)2 * WM * WN * WK) * LOOP * BATCH_SIZE / (ms / 1000) * 1e-12;
    return tflops;
}

int main() {
    std::cout << "==== wmma_acc_test ====\n";
    wmma_test<1, 4, 256, true>();
    std::cout << "==== wmma_perf_test ====\n";
    constexpr int LOOP = 1000000;
    constexpr int NBLOCKS = 4096;
    constexpr int BLOCK_SIZE = 256;
    for (int i = 0; i < 3; i++) {
        auto tflops = wmma_test<LOOP, NBLOCKS, BLOCK_SIZE, false>();
        std::cout << tflops << " TFLOPS" << std::endl;
    }
    return 0;
}
