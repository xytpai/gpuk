#pragma once

#include "device_common.h"
#include "negzero.h"

namespace kernel_utils {

template <typename T, int WARP_SIZE, typename func_t>
__device__ __forceinline__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val = fn(val, __shfl_xor(val, offset, WARP_SIZE));
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
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] =
                __builtin_nontemporal_load((uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t *>(&data)[i],
                                        (uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void volatile_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] = __scoped_atomic_load_n(
                (uint32_t *)ptr + i, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void volatile_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __scoped_atomic_store_n((uint32_t *)ptr + i,
                                    reinterpret_cast<uint32_t *>(&data)[i],
                                    __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = val;
        }
    }
    template <typename VT>
    __device__ __forceinline__ void cast_fill(VT val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            *reinterpret_cast<VT *>(&data[i]) = val;
        }
    }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE> &self,
                                         const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[i] = (float)self[i] + (float)other[i];
    }
}

template <typename T, int VEC_SIZE, int NRanks>
__device__ __forceinline__ void vec_add_r_(vec_t<T, VEC_SIZE> (&self)[NRanks]) {
    vec_t<float, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] = (float)self[0][i];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            acc[i] += (float)self[r][i];
        }
    }
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[0][i] = (T)acc[i];
    }
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (is_negative_zero<T>(vec[i])) {
            return true;
        }
    }
    return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        vec[i] = (is_negative_zero<T>(vec[i])) ? static_cast<T>(0.f) : vec[i];
    }
}

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

} // namespace kernel_utils
