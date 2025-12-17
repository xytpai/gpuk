#include <cassert>
#include <iostream>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gpuSuccess hipSuccess
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#endif

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#define gpuSuccess cudaSuccess
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#endif

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename T, int vec_size>
__global__ void threads_copy_kernel(T *in, T *out) {
    using vec_t = aligned_array<T, vec_size>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    reinterpret_cast<vec_t *>(out)[idx] = reinterpret_cast<vec_t *>(in)[idx];
}

template <typename T, int vec_size>
float threads_copy(T *in, T *out, int n, gpuStream_t s) {
    int block_size = 128;
    int block_work_size = block_size * vec_size;
    assert(n % block_work_size == 0);
    int nblocks = n / block_work_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(nblocks);

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);

    threads_copy_kernel<T, vec_size><<<numBlocks, threadsPerBlock, 0, s>>>(in, out);
    gpuDeviceSynchronize();

    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);
    return ms;
}

template <typename scalar_t, int vec_size>
void test_threads_copy(int n) {
    auto in_cpu = new scalar_t[n];
    auto out_cpu = new scalar_t[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    scalar_t *in_gpu, *out_gpu;
    gpuMalloc(&in_gpu, n * sizeof(scalar_t));
    gpuMalloc(&out_gpu, n * sizeof(scalar_t));
    gpuMemcpy(in_gpu, in_cpu, n * sizeof(scalar_t), gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();

    float timems;
    for (int i = 0; i < 3; i++)
        timems = threads_copy<scalar_t, vec_size>(in_gpu, out_gpu, n, 0);
    std::cout << "timeus:" << timems * 1000 << " throughput:";

    float total_GBytes = (n + n) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS val:";

    gpuMemcpy(out_cpu, out_gpu, n * sizeof(scalar_t), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        auto diff = (float)out_cpu[i] - (float)in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error: " << "ref:" << (float)in_cpu[i] << ", actual:" << (float)out_cpu[i] << "\n";
            return;
        }
    }
    std::cout << "ok\n";

    gpuFree(in_gpu);
    gpuFree(out_gpu);
    delete[] in_cpu;
    delete[] out_cpu;
}

int main() {
    constexpr int vec_size = 4;
    int n;
    n = 128 * vec_size;
    std::cout << n << " bytes small copy kernel test ...\n";
    test_threads_copy<float, vec_size>(n);
    test_threads_copy<float, vec_size>(n);
    n = 256 * vec_size;
    std::cout << n << " bytes small copy kernel test ...\n";
    test_threads_copy<float, vec_size>(n);
    test_threads_copy<float, vec_size>(n);
}
