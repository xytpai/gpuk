#include "device_common.h"

template <int LOOP>
__global__ void fmad_loop_kernel(float *x) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float a = x[index], b = -1.0f;
    for (int i = 0; i < LOOP; i++) {
        for (int j = 0; j < LOOP; j++) {
            a = a * b + b;
        }
    }
    x[index] = a;
}

template <int LOOP, int block_size, int num_blocks>
float fmad_test() {
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(num_blocks);
    constexpr int n = block_size * num_blocks;
    auto x = new float[n];
    float *dx;
    gpuMalloc(&dx, n * sizeof(float));
    gpuMemcpy(dx, x, n * sizeof(float), gpuMemcpyHostToDevice);

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);

    fmad_loop_kernel<LOOP><<<numBlocks, threadsPerBlock>>>(dx);
    gpuDeviceSynchronize();

    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);

    gpuMemcpy(x, dx, n * sizeof(float), gpuMemcpyDeviceToHost);

    gpuFree(dx);
    delete[] x;
    return ms;
}

int main() {
    constexpr int LOOP = 10000;
    constexpr int block_size = 256;
    constexpr int num_blocks = 2048;
    for (int i = 0; i < 3; i++) {
        auto timems = fmad_test<LOOP, block_size, num_blocks>();
        auto tflops =
            2.0 * LOOP * LOOP * num_blocks * block_size / (timems / 1000) * 1e-12;
        auto arithmetic_intensity = 2.0f * LOOP * LOOP / (sizeof(float) * 2);
        std::cout << "arithmetic_intensity: " << arithmetic_intensity << " FLOP/Byte.  |  COMPUTE:";
        std::cout << tflops << " TFLOPS" << std::endl;
    }
    return 0;
}
