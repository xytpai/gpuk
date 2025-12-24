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
#define gpuGraph_t hipGraph_t
#define gpuGraphExec_t hipGraphExec_t
#define gpuStreamBeginCapture hipStreamBeginCapture
#define gpuStreamEndCapture hipStreamEndCapture
#define gpuStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define gpuGraphInstantiate hipGraphInstantiate
#define gpuGraphLaunch hipGraphLaunch
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
#define gpuGraph_t cudaGraph_t
#define gpuGraphExec_t cudaGraphExec_t
#define gpuStreamBeginCapture cudaStreamBeginCapture
#define gpuStreamEndCapture cudaStreamEndCapture
#define gpuStreamCaptureModeGlobal cudaStreamCaptureModeGlobal
#define gpuGraphInstantiate cudaGraphInstantiate
#define gpuGraphLaunch cudaGraphLaunch
#endif

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename T, int vec_size>
__global__ void threads_inc_kernel(T *in) {
    using vec_t = aligned_array<T, vec_size>;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto vec = reinterpret_cast<vec_t *>(in)[idx];
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
        vec.val[i] += 1;
    }
    reinterpret_cast<vec_t *>(in)[idx] = vec;
}

template <typename T, int vec_size>
float threads_inc(T *in, int n, gpuStream_t stream, int LOOP) {
    int block_size = 256;
    int block_work_size = block_size * vec_size;
    assert(n % block_work_size == 0);
    int nblocks = n / block_work_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(nblocks);

    gpuGraph_t graph;
    gpuGraphExec_t exec;
    gpuStreamBeginCapture(stream, gpuStreamCaptureModeGlobal);
    for (int i = 0; i < LOOP; ++i) {
        threads_inc_kernel<T, vec_size><<<numBlocks, threadsPerBlock, 0, stream>>>(in + i * n);
    }
    gpuStreamEndCapture(stream, &graph);
    gpuGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    float ms = 0;
    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start, stream);
    gpuGraphLaunch(exec, stream);
    gpuEventRecord(stop, stream);
    gpuEventSynchronize(stop);
    gpuEventElapsedTime(&ms, start, stop);
    ms /= LOOP;
    return ms;
}

template <typename scalar_t, int vec_size>
void test_threads_inc(int n, int LOOP) {
    constexpr int FLUSH_SIZE = 1024 * 1024 * 1024;
    auto in_cpu = new scalar_t[LOOP * n];
    auto flush_data = new char[FLUSH_SIZE];
    memset(flush_data, 1, FLUSH_SIZE);
    for (int i = 0; i < LOOP * n; ++i) {
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    scalar_t *in_gpu, *flush_ptr;
    gpuMalloc(&in_gpu, LOOP * n * sizeof(scalar_t));
    gpuMalloc(&flush_ptr, FLUSH_SIZE);
    gpuMemcpy(in_gpu, in_cpu, LOOP * n * sizeof(scalar_t), gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();
    gpuMemcpy(flush_ptr, flush_data, FLUSH_SIZE, gpuMemcpyHostToDevice);
    gpuDeviceSynchronize();

    gpuStream_t stream;
    gpuStreamCreate(&stream);

    float timems = threads_inc<scalar_t, vec_size>(in_gpu, n, stream, LOOP);
    std::cout << "timeus:" << timems * 1000 << " throughput:";

    float total_GBytes = (n + n) * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS val:";

    auto out_cpu = new scalar_t[LOOP * n];
    gpuMemcpy(out_cpu, in_gpu, LOOP * n * sizeof(scalar_t), gpuMemcpyDeviceToHost);
    gpuDeviceSynchronize();

    for (int i = 0; i < LOOP * n; i++) {
        float out = (float)out_cpu[i];
        float ref = (float)in_cpu[i] + 1;
        auto diff = out - ref;
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error: "
                      << "ref:" << ref << ", actual:" << out << "\n";
            return;
        }
    }
    std::cout << "ok\n";

    gpuFree(in_gpu);
    gpuFree(flush_ptr);
    delete[] in_cpu;
    delete[] out_cpu;
    delete[] flush_data;
}

int main(int argc, char **argv) {
    constexpr int vec_size = 4;
    int n;
    n = 256 * vec_size;
    int LOOP = std::stoi(argv[1]);
    std::cout << n * sizeof(float) << " bytes small inc kernel test with LOOP=" << LOOP << " ...\n";
    test_threads_inc<float, vec_size>(n, LOOP);
    test_threads_inc<float, vec_size>(n, LOOP);
}
