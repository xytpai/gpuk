#pragma once

#include "device_common.h"
#include "negzero.h"

namespace kernel_utils {

static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;

template <typename T, int WARP_SIZE>
__device__ __forceinline__ T warp_shfl_sync(T val, int src_id) {
#ifdef __CUDACC__
    return __shfl_sync(0xffffffff, val, src_id, WARP_SIZE);
#elif defined(__HIPCC__)
    return __shfl(val, src_id, WARP_SIZE);
#endif
}

template <typename T, int WARP_SIZE>
__device__ __forceinline__ T warp_shfl_xor_sync(T val, int offset) {
#ifdef __CUDACC__
    return __shfl_xor_sync(0xffffffff, val, offset, WARP_SIZE);
#elif defined(__HIPCC__)
    return __shfl_xor(val, offset, WARP_SIZE);
#endif
}

template <typename T, int WARP_SIZE, typename func_t>
__device__ __forceinline__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val = fn(val, warp_shfl_xor_sync<T, WARP_SIZE>(val, offset));
    }
    return val;
}

template <typename T, int WARP_SIZE, int BLOCK_SIZE, typename func_t>
__device__ __forceinline__ T block_reduce(T val, func_t fn) {
    static __shared__ T shared[BLOCK_SIZE / WARP_SIZE];
    const int tid = threadIdx.x;
    const int w_tid = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    val = warp_reduce<T, WARP_SIZE, func_t>(val, fn);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = shared[w_tid];
    __syncthreads();
    val = warp_reduce<T, BLOCK_SIZE / WARP_SIZE, func_t>(val, fn);
    return val;
}

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) vec_t {
    T data[VEC_SIZE];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(const T *ptr) {
        *this = *reinterpret_cast<vec_t<T, VEC_SIZE> *>(const_cast<T *>(ptr));
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, VEC_SIZE> *>(ptr) = *this;
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = val;
        }
    }
    __device__ __forceinline__ void copy_(const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = other.data[i];
        }
    }
    template <typename IT>
    __device__ __forceinline__ void from_(const vec_t<IT, VEC_SIZE> &src, float scale) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            if constexpr (std::is_same_v<T, IT>) {
                data[i] = src[i];
            } else {
                data[i] = static_cast<T>((float)(src[i]) / scale);
            }
        }
    }
    __device__ __forceinline__ void add_(const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] += other.data[i];
        }
    }
};

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

struct CopyAsync {
    template <typename T>
    static __device__ __forceinline__ void add(T *dst, T *src) {
#ifdef __CUDACC__
        constexpr int NBYTES = sizeof(T);
        auto dst_ = (uint32_t)(__cvta_generic_to_shared(dst));
        auto src_ = reinterpret_cast<uint64_t>(src);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_), "l"(src_), "n"(NBYTES));
#else
        *dst = *src;
#endif
    }
    static __device__ __forceinline__ void commit() {
#ifdef __CUDACC__
        asm volatile("cp.async.commit_group;\n" ::);
#endif
    }
    template <int S = 0>
    static __device__ __forceinline__ void wait() {
#ifdef __CUDACC__
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S));
#endif
    }
};

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

} // namespace kernel_utils
