#pragma once

#include "device_common.h"

using namespace kernel_utils;

namespace sgemm {

template <typename scalar_t>
__global__ void sgemm_naive_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    int mi = blockIdx.y * 32 + threadIdx.y;
    int ni = blockIdx.x * 32 + threadIdx.x;
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += a[mi * k + ki] * b[ki * n + ni];
        }
        out[mi * n + ni] = alpha * acc + beta * out[mi * n + ni];
    }
}

template <typename scalar_t>
void sgemm_naive(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta,
    gpuStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    sgemm_naive_kernel<scalar_t><<<grid, block, 0, stream>>>(out, a, b, m, n, k, alpha, beta);
}

template <typename scalar_t, int VEC_A, int VEC_B>
struct mma_reg_t {
    using vec_a_t = aligned_array<scalar_t, VEC_A>;
    using vec_b_t = aligned_array<scalar_t, VEC_B>;
    union {
        vec_a_t a_vec;
        scalar_t a[VEC_A];
    };
    union {
        vec_b_t b_vec;
        scalar_t b[VEC_B];
    };
};

template <
    typename scalar_t,
    int BLOCK_K,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N,
    int KSTRIDE_A,
    int KSTRIDE_B>
struct WarpTile {
    __device__ __forceinline__ void operator()(scalar_t *o, scalar_t *a, scalar_t *b, int wy, int wx, int w_tid) {
        using a_vec_t = aligned_array<scalar_t, VEC_M>;
        using b_vec_t = aligned_array<scalar_t, VEC_N>;
        int th_y = wy + w_tid / WARP_N_THREADS * VEC_M;
        int th_x = wx + w_tid % WARP_N_THREADS * VEC_N;
        mma_reg_t<scalar_t, VEC_M, VEC_N> reg;
#pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            reg.a_vec = *reinterpret_cast<a_vec_t *>(&a[k * KSTRIDE_A + th_y]);
            reg.b_vec = *reinterpret_cast<b_vec_t *>(&b[k * KSTRIDE_B + th_x]);
#pragma unroll
            for (int i = 0; i < VEC_M; ++i) {
#pragma unroll
                for (int j = 0; j < VEC_N; ++j) {
                    o[i * VEC_N + j] += reg.a[i] * reg.b[j];
                }
            }
        }
    }
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N>
struct BlockTile {
    enum {
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * 32,
        WARP_ATOM_M = WARP_M_THREADS * VEC_M,
        WARP_ATOM_N = WARP_N_THREADS * VEC_N,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        BLOCK_M = BLOCK_M_WARPS * WARP_M,
        BLOCK_N = BLOCK_N_WARPS * WARP_N,
        BLOCK_KM_SIZE = BLOCK_K * BLOCK_M,
        BLOCK_KN_SIZE = BLOCK_K * BLOCK_N,
        LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_B_X_THREADS = BLOCK_N / LDG_VEC_SIZE,
        LDG_REG_A_COUNT = BLOCK_KM_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        LDG_REG_B_COUNT = BLOCK_KN_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        APAD = 4, // swizzle is not a good idea for sgemm
    };
    static_assert(WARP_M_THREADS * WARP_N_THREADS == 32);
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;

    __device__ __forceinline__ BlockTile(int tid) :
        tid(tid), wid(tid >> 5), w_tid(tid & 31),
        ldg_a_vec_idx(tid % LDG_A_X_THREADS),
        ldg_b_vec_idx(tid % LDG_B_X_THREADS) {
    }

    __device__ __forceinline__ void ldg(const scalar_t *a, int a_stride, const scalar_t *b, int b_stride) {
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++)
            ldg_a_reg[i] = reinterpret_cast<ldg_vec_t *>(
                const_cast<scalar_t *>(a) + ((BLOCK_THREADS * i + tid) / LDG_A_X_THREADS) * a_stride)[ldg_a_vec_idx];
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++)
            ldg_b_reg[i] = reinterpret_cast<ldg_vec_t *>(
                const_cast<scalar_t *>(b) + ((BLOCK_THREADS * i + tid) / LDG_B_X_THREADS) * b_stride)[ldg_b_vec_idx];
    }

    __device__ __forceinline__ void sts(scalar_t *as, scalar_t *bs) {
        auto bs_vec = reinterpret_cast<ldg_vec_t *>(bs);
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            int y = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
#pragma unroll
            for (int j = 0; j < LDG_VEC_SIZE; j++) {
                as[(ldg_a_vec_idx * LDG_VEC_SIZE + j) * BLOCK_M + y] = ldg_a_reg[i].val[j];
            }
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            bs_vec[BLOCK_THREADS * i + tid] = ldg_b_reg[i];
        }
    }

    __device__ __forceinline__ void ldg_copy_async(
        scalar_t *as, scalar_t *bs,
        const scalar_t *a, int a_stride, const scalar_t *b, int b_stride) {
        as_ = as;
        a_ = a;
        a_stride_ = a_stride;
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            CopyAsync::add(
                &(reinterpret_cast<ldg_vec_t *>(bs)[BLOCK_THREADS * i + tid]),
                &(reinterpret_cast<ldg_vec_t *>(
                    const_cast<scalar_t *>(b) + ((BLOCK_THREADS * i + tid) / LDG_B_X_THREADS) * b_stride)[ldg_b_vec_idx]));
        }
        CopyAsync::commit();
    }

    __device__ __forceinline__ void convert_layout() {
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++)
            ldg_a_reg[i] = reinterpret_cast<ldg_vec_t *>(
                const_cast<scalar_t *>(a_) + ((BLOCK_THREADS * i + tid) / LDG_A_X_THREADS) * a_stride_)[ldg_a_vec_idx];
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            int y = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
#pragma unroll
            for (int j = 0; j < LDG_VEC_SIZE; j++) {
                int x = ldg_a_vec_idx * LDG_VEC_SIZE + j;
                as_[x * (BLOCK_M + APAD) + y] = ldg_a_reg[i].val[j];
            }
        }
        CopyAsync::wait();
    }

    __device__ __forceinline__ void operator()(scalar_t (*o)[VEC_M * VEC_N], scalar_t *a, scalar_t *b) {
        int warp_y = wid / BLOCK_N_WARPS * WARP_M;
        int warp_x = wid % BLOCK_N_WARPS * WARP_N;
        WarpTile<scalar_t, BLOCK_K, WARP_M_THREADS, WARP_N_THREADS, VEC_M, VEC_N, BLOCK_M + APAD, BLOCK_N> warp_tile;
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
            int warp_atom_offset_y = warp_y + i * WARP_ATOM_M;
#pragma unroll
            for (int j = 0; j < WARP_N_STEPS; ++j) {
                int warp_atom_offset_x = warp_x + j * WARP_ATOM_N;
                warp_tile(o[i * WARP_N_STEPS + j], a, b, warp_atom_offset_y, warp_atom_offset_x, w_tid);
            }
        }
    }

private:
    int tid;
    int wid;
    int w_tid;
    int ldg_a_vec_idx;
    int ldg_b_vec_idx;
    ldg_vec_t ldg_a_reg[LDG_REG_A_COUNT];
    ldg_vec_t ldg_b_reg[LDG_REG_B_COUNT];
    scalar_t *as_;
    const scalar_t *a_;
    int a_stride_;
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N>
__global__ void sgemm_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    constexpr int WARP_M = WARP_M_STEPS * WARP_M_THREADS * VEC_M;
    constexpr int WARP_N = WARP_N_STEPS * WARP_N_THREADS * VEC_N;
    constexpr int BLOCK_M = BLOCK_M_WARPS * WARP_M;
    constexpr int BLOCK_N = BLOCK_N_WARPS * WARP_N;

    // get idx
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int w_tid = tid & 31;
    int block_y = blockIdx.y;
    int block_x = blockIdx.z * gridDim.x + blockIdx.x;

    // get slm
    using BlockTileT = BlockTile<scalar_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS,
                                 WARP_M_STEPS, WARP_N_STEPS, WARP_M_THREADS, WARP_N_THREADS, VEC_M, VEC_N>;
    constexpr int SLM_SIZE = BLOCK_K * (BLOCK_M + BlockTileT::APAD + BLOCK_N) * 2;
    constexpr int BLOCK_KM_SIZE = BLOCK_K * (BLOCK_M + BlockTileT::APAD);
    constexpr int BLOCK_KN_SIZE = BLOCK_K * BLOCK_N;
    __shared__ scalar_t slm[SLM_SIZE];
    scalar_t *as = &slm[0];
    scalar_t *bs = as + BLOCK_KM_SIZE * 2;

    // init regs
    scalar_t o_reg[WARP_M_STEPS * WARP_N_STEPS][VEC_M * VEC_N] = {{(scalar_t)0}};
    BlockTileT block_tile(tid);
    int current_stage = 0;
    int next_stage = 1;
    int a_begin = block_y * BLOCK_M * k;
    int b_begin = block_x * BLOCK_N;
    int a_end = block_y * BLOCK_M * k + k;

    block_tile.ldg_copy_async(
        &as[current_stage * BLOCK_KM_SIZE], &bs[current_stage * BLOCK_KN_SIZE],
        &a[a_begin], k, &b[b_begin], n);
    block_tile.convert_layout();
    __syncthreads();

    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K * n) {
        bool has_next = a_begin + BLOCK_K < a_end;
        if (has_next) {
            block_tile.ldg_copy_async(
                &as[next_stage * BLOCK_KM_SIZE], &bs[next_stage * BLOCK_KN_SIZE],
                &a[a_begin + BLOCK_K], k, &b[b_begin + BLOCK_K * n], n);
        }
        block_tile(o_reg, &as[current_stage * BLOCK_KM_SIZE], &bs[current_stage * BLOCK_KN_SIZE]);
        if (has_next) {
            block_tile.convert_layout();
        }
        __syncthreads();
        current_stage ^= 1;
        next_stage ^= 1;
    }

    { // write back
        using stg_vec_t = aligned_array<scalar_t, VEC_N>;
        int out_warp_y = block_y * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        int out_warp_x = block_x * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
        constexpr int WARP_ATOM_M = WARP_M / WARP_M_STEPS;
        constexpr int WARP_ATOM_N = WARP_N / WARP_N_STEPS;
#pragma unroll
        for (int lm = 0; lm < WARP_M_STEPS; lm++) {
#pragma unroll
            for (int ln = 0; ln < WARP_N_STEPS; ln++) {
                int out_thread_y = out_warp_y + lm * WARP_ATOM_M + w_tid / WARP_N_THREADS * VEC_M;
                int out_thread_x = out_warp_x + ln * WARP_ATOM_N + w_tid % WARP_N_THREADS * VEC_N;
#pragma unroll
                for (int i = 0; i < VEC_M; i++) {
                    int y = out_thread_y + i;
                    if (y < m && out_thread_x < n) {
                        auto vec = *reinterpret_cast<stg_vec_t *>(out + y * n + out_thread_x);
#pragma unroll
                        for (int j = 0; j < VEC_N; j++) {
                            vec.val[j] = alpha * o_reg[lm * WARP_N_STEPS + ln][i * VEC_N + j] + beta * vec.val[j];
                        }
                        *reinterpret_cast<stg_vec_t *>(out + y * n + out_thread_x) = vec;
                    }
                }
            }
        }
    }
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void sgemm_(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    assert(m % VEC_SIZE == 0);
    assert(n % VEC_SIZE == 0);
    assert(k % VEC_SIZE == 0);
    int m_blocks = (m + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (n + BLOCK_N - 1) / BLOCK_N;
    int split_num = (n_blocks + 128 - 1) / 128;
    dim3 grid((n_blocks + split_num - 1) / split_num, m_blocks, split_num);
    if constexpr (BLOCK_M == 64 && BLOCK_N == 64) {
        dim3 block(256);
        constexpr int BLOCK_K = 32;
        sgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                     /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_STEPS*/ 1, /*WARP_N_STEPS*/ 1,
                     /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 32 && BLOCK_N == 64) {
        dim3 block(128);
        constexpr int BLOCK_K = 32;
        sgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                     /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_M_STEPS*/ 1, /*WARP_N_STEPS*/ 1,
                     /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 128 && BLOCK_N == 128) {
        dim3 block(256);
        constexpr int BLOCK_K = 16;
        sgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                     /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_M_STEPS*/ 2, /*WARP_N_STEPS*/ 2,
                     /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 64 && BLOCK_N == 128) {
        dim3 block(128);
        constexpr int BLOCK_K = 16;
        sgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K,
                     /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_M_STEPS*/ 2, /*WARP_N_STEPS*/ 2,
                     /*WARP_M_THREADS*/ 4, /*WARP_N_THREADS*/ 8, /*VEC_M*/ 4, /*VEC_N*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else {
        assert(false);
    }
}

template <typename scalar_t>
void sgemm(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta,
    gpuStream_t stream) {
    auto min_size = std::min(m, n);
    if (min_size <= 512) {
        sgemm_<scalar_t, 64, 64>(out, a, b, m, n, k, alpha, beta, stream);
    } else {
        sgemm_<scalar_t, 64, 128>(out, a, b, m, n, k, alpha, beta, stream);
    }
}

} // namespace sgemm
