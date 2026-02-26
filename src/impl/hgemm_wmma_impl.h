#pragma once

#include "device_common.h"

using namespace kernel_utils;

namespace hgemm {

template <typename scalar_t, typename acc_t>
struct WMMA_M16N8K16 {
    enum {
        M = 16,
        N = 8,
        K = 16,
    };
    using FragmentAT = aligned_array<scalar_t, 8>;
    using FragmentBT = aligned_array<scalar_t, 4>;
    using FragmentCT = aligned_array<acc_t, 4>;

    __device__ __forceinline__ WMMA_M16N8K16() {
        w_tid = threadIdx.x & 31;
    }

    __device__ __forceinline__ void operator()(
        FragmentCT &d,
        FragmentAT const &a,
        FragmentBT const &b,
        FragmentCT const &c) {
#ifdef __CUDACC__
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        acc_t const *C = reinterpret_cast<acc_t const *>(&c);
        acc_t *D = reinterpret_cast<acc_t *>(&d);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
    }

    __device__ __forceinline__ void reset_fragment_c(FragmentCT &c) {
        c.val[0] = 0;
        c.val[1] = 0;
        c.val[2] = 0;
        c.val[3] = 0;
    }

    template <uint32_t VEC_BITS = 3>
    __device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
        constexpr uint32_t COL_BITS = 7 - 4; // 32*4B (7bits) - 16B (4bits)
        constexpr uint32_t COL_MASK = ((1 << COL_BITS) - 1) << VEC_BITS;
        return ((addr >> VEC_BITS) & COL_MASK) ^ addr;
    }

    __device__ __forceinline__ void load_matrix_a(FragmentAT &a, scalar_t *base_ptr, int offset, int stride) {
#ifdef __CUDACC__
        auto A = reinterpret_cast<uint32_t *>(&a);
        uint32_t offset_ = swizzle(offset + (w_tid % 16) * stride + (w_tid / 16) * 8);
        auto addr = (uint32_t)__cvta_generic_to_shared(base_ptr + offset_);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
            : "r"(addr));
#endif
    }

    __device__ __forceinline__ void load_matrix_b(FragmentBT &b, scalar_t *base_ptr, int offset, int stride) {
#ifdef __CUDACC__
        auto B = reinterpret_cast<uint32_t *>(&b);
        uint32_t offset_ = swizzle(offset + (w_tid % 16) * stride);
        auto addr = (uint32_t)__cvta_generic_to_shared(base_ptr + offset_);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(B[0]), "=r"(B[1])
            : "r"(addr));
#endif
    }

    __device__ __forceinline__ void store_matrix(scalar_t *ptr, int stride, FragmentCT const &c, acc_t alpha, acc_t beta) {
        auto y = w_tid / 4;
        auto x = w_tid % 4 * 2;
        using vec_t = aligned_array<scalar_t, 2>;
        auto vec0 = *reinterpret_cast<vec_t *>(&ptr[y * stride + x]);
        auto vec1 = *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]);
        vec0.val[0] = alpha * (acc_t)c.val[0] + beta * (acc_t)vec0.val[0];
        vec0.val[1] = alpha * (acc_t)c.val[1] + beta * (acc_t)vec0.val[1];
        vec1.val[0] = alpha * (acc_t)c.val[2] + beta * (acc_t)vec1.val[0];
        vec1.val[1] = alpha * (acc_t)c.val[3] + beta * (acc_t)vec1.val[1];
        *reinterpret_cast<vec_t *>(&ptr[y * stride + x]) = vec0;
        *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]) = vec1;
    }

private:
    int w_tid;
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS>
struct BlockTile {
    using WMMAT = WMMA_M16N8K16<scalar_t, float>;
    using FragmentAT = typename WMMAT::FragmentAT;
    using FragmentBT = typename WMMAT::FragmentBT;
    using FragmentCT = typename WMMAT::FragmentCT;
    enum {
        WARP_ATOM_M = WMMAT::M,
        WARP_ATOM_N = WMMAT::N,
        WARP_ATOM_K = WMMAT::K,
        WARP_K_STEPS = BLOCK_K / WARP_ATOM_K,
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * 32,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        BLOCK_M = BLOCK_M_WARPS * WARP_M,
        BLOCK_N = BLOCK_N_WARPS * WARP_N,
        BLOCK_MK_SIZE = BLOCK_M * BLOCK_K,
        BLOCK_KN_SIZE = BLOCK_K * BLOCK_N,
        LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_B_X_THREADS = BLOCK_N / LDG_VEC_SIZE,
        LDG_REG_A_COUNT = BLOCK_MK_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        LDG_REG_B_COUNT = BLOCK_KN_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
    };
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;

    __device__ __forceinline__ BlockTile(int tid) :
        tid(tid), wid(tid >> 5), w_tid(tid & 31),
        ldg_a_vec_idx(tid % LDG_A_X_THREADS),
        ldg_b_vec_idx(tid % LDG_B_X_THREADS) {
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_N_STEPS; ++j) {
                wmma.reset_fragment_c(fo[i][j]);
            }
        }
    }

    __device__ __forceinline__ void ldg_copy_async(
        scalar_t *as, scalar_t *bs,
        const scalar_t *a, int a_stride, const scalar_t *b, int b_stride) {
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            auto offset = wmma.swizzle((BLOCK_THREADS * i + tid) * LDG_VEC_SIZE);
            CopyAsync::add(
                reinterpret_cast<ldg_vec_t *>(as + offset),
                &(reinterpret_cast<ldg_vec_t *>(
                    const_cast<scalar_t *>(a) + ((BLOCK_THREADS * i + tid) / LDG_A_X_THREADS) * a_stride)[ldg_a_vec_idx]));
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            auto offset = wmma.swizzle((BLOCK_THREADS * i + tid) * LDG_VEC_SIZE);
            CopyAsync::add(
                reinterpret_cast<ldg_vec_t *>(bs + offset),
                &(reinterpret_cast<ldg_vec_t *>(
                    const_cast<scalar_t *>(b) + ((BLOCK_THREADS * i + tid) / LDG_B_X_THREADS) * b_stride)[ldg_b_vec_idx]));
        }
    }

    template <int S = 0>
    __device__ __forceinline__ void wait() {
        CopyAsync::wait<S>();
    }

    __device__ __forceinline__ void commit() {
        CopyAsync::commit();
    }

    __device__ __forceinline__ void load_matrix(scalar_t *as, scalar_t *bs) {
        int warp_y = wid / BLOCK_N_WARPS * WARP_M;
        int warp_x = wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
            int warp_atom_offset_y = warp_y + i * WARP_ATOM_M;
#pragma unroll
            for (int k = 0; k < WARP_K_STEPS; ++k) {
                int offset = warp_atom_offset_y * BLOCK_K + k * WARP_ATOM_K;
                wmma.load_matrix_a(fa[k][i], as, offset, BLOCK_K);
            }
        }
#pragma unroll
        for (int j = 0; j < WARP_N_STEPS; ++j) {
            int warp_atom_offset_x = warp_x + j * WARP_ATOM_N;
#pragma unroll
            for (int k = 0; k < WARP_K_STEPS; ++k) {
                int offset = warp_atom_offset_x + k * WARP_ATOM_K * BLOCK_N;
                wmma.load_matrix_b(fb[k][j], bs, offset, BLOCK_N);
            }
        }
    }

    __device__ __forceinline__ void store_matrix(scalar_t *ptr, int by, int bx, int m, int n, float alpha, float beta) {
        int warp_y = by * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        int warp_x = bx * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
            int warp_atom_offset_y = warp_y + i * WARP_ATOM_M;
#pragma unroll
            for (int j = 0; j < WARP_N_STEPS; ++j) {
                int warp_atom_offset_x = warp_x + j * WARP_ATOM_N;
                if (warp_atom_offset_y < m && warp_atom_offset_x < n) {
                    auto ptr_ = ptr + warp_atom_offset_y * n + warp_atom_offset_x;
                    wmma.store_matrix(ptr_, n, fo[i][j], alpha, beta);
                    __syncthreads();
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() {
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_N_STEPS; ++j) {
#pragma unroll
                for (int k = 0; k < WARP_K_STEPS; ++k) {
                    wmma(fo[i][j], fa[k][i], fb[k][j], fo[i][j]);
                }
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
    WMMAT wmma;
    FragmentAT fa[WARP_K_STEPS][WARP_M_STEPS];
    FragmentBT fb[WARP_K_STEPS][WARP_N_STEPS];
    FragmentCT fo[WARP_M_STEPS][WARP_N_STEPS];
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS,
    int STAGES = 3>
__global__ void hgemm_kernel(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const float alpha,
    const float beta) {
    using BlockTileT = BlockTile<scalar_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS>;
    constexpr int BLOCK_M = BlockTileT::BLOCK_M;
    constexpr int BLOCK_N = BlockTileT::BLOCK_N;

    // get idx
    int tid = threadIdx.x;
    int block_y = blockIdx.y;
    int block_x = blockIdx.z * gridDim.x + blockIdx.x;

    // get slm
    __shared__ scalar_t as[STAGES][BLOCK_M * BLOCK_K];
    __shared__ scalar_t bs[STAGES][BLOCK_K * BLOCK_N];

    // init regs
    BlockTileT block_tile(tid);
    int current_stage = 0;
    int a_begin = block_y * BLOCK_M * k;
    int b_begin = block_x * BLOCK_N;
    int a_end = a_begin + k;

#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        block_tile.ldg_copy_async(as[s], bs[s], &a[a_begin + s * BLOCK_K], k, &b[b_begin + s * BLOCK_K * n], n);
        block_tile.commit();
    }
    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K * n) {
        block_tile.template wait<STAGES - 1>();
        __syncthreads();
        block_tile.load_matrix(as[current_stage], bs[current_stage]);
        block_tile();
        __syncthreads();
        if (a_begin + STAGES * BLOCK_K < a_end) {
            block_tile.ldg_copy_async(as[current_stage], bs[current_stage],
                                      &a[a_begin + STAGES * BLOCK_K], k, &b[b_begin + STAGES * BLOCK_K * n], n);
        }
        block_tile.commit();
        current_stage = (current_stage + 1) % STAGES;
    }

    block_tile.store_matrix(out, block_y, block_x, m, n, alpha, beta);
}

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
void hgemm_(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const float alpha,
    const float beta,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    assert(m % VEC_SIZE == 0);
    assert(n % VEC_SIZE == 0);
    assert(k % VEC_SIZE == 0);
    int m_blocks = (m + BLOCK_M - 1) / BLOCK_M;
    int n_blocks = (n + BLOCK_N - 1) / BLOCK_N;
    int split_num = (n_blocks + 32 - 1) / 32;
    dim3 grid((n_blocks + split_num - 1) / split_num, m_blocks, split_num);
    if constexpr (BLOCK_M == 64 && BLOCK_N == 32) {
        dim3 block(128);
        constexpr int BLOCK_K = 32;
        hgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2,
                     /*WARP_M_STEPS*/ 2, /*WARP_N_STEPS*/ 2><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 64 && BLOCK_N == 64) {
        dim3 block(128);
        constexpr int BLOCK_K = 32;
        hgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2,
                     /*WARP_M_STEPS*/ 2, /*WARP_N_STEPS*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else if constexpr (BLOCK_M == 128 && BLOCK_N == 128) {
        dim3 block(256);
        constexpr int BLOCK_K = 16;
        hgemm_kernel<scalar_t, /*BLOCK_K*/ BLOCK_K, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4,
                     /*WARP_M_STEPS*/ 4, /*WARP_N_STEPS*/ 4><<<grid, block, 0, stream>>>(
            out, a, b, m, n, k, alpha, beta);
    } else {
        assert(false);
    }
}

template <typename scalar_t>
void hgemm(
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const float alpha,
    const float beta,
    gpuStream_t stream) {
    auto min_size = std::min(m, n);
    if (min_size <= 2048) {
        hgemm_<scalar_t, 64, 64>(out, a, b, m, n, k, alpha, beta, stream);
    } else {
        hgemm_<scalar_t, 128, 128>(out, a, b, m, n, k, alpha, beta, stream);
    }
}

} // namespace hgemm
