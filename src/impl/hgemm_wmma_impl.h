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
        acc_t *D = reinterpret_cast<acc_t *>(d);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif
    }

    __device__ __forceinline__ void fill_fragment_c(FragmentCT &c) {
        c.val[0] = val;
        c.val[1] = val;
        c.val[2] = val;
        c.val[3] = val;
    }

    __device__ __forceinline__ void load_matrix_a(FragmentAT &a, int stride) {
#ifdef __CUDACC__
        auto A = reinterpret_cast<uint32_t *>(&a);
        auto addr = (uint32_t)__cvta_generic_to_shared(ptr + (w_tid % 16) * stride + (w_tid / 16) * 8);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
            : "r"(addr));
#endif
    }

    __device__ __forceinline__ void load_matrix_b(FragmentBT &b, int stride) {
#ifdef __CUDACC__
        auto B = reinterpret_cast<uint32_t *>(&b);
        auto addr = (uint32_t)__cvta_generic_to_shared(ptr + (w_tid % 16) * stride);
        asm volatile(
            "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(B[0]), "=r"(B[1])
            : "r"(addr));
#endif
    }

    __device__ __forceinline__ void store_matrix(acc_t *ptr, int stride, FragmentCT const &c, acc_t alpha, acc_t beta) {
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
        auto as_vec = reinterpret_cast<ldg_vec_t *>(as);
        auto bs_vec = reinterpret_cast<ldg_vec_t *>(bs);
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            as_vec[BLOCK_THREADS * i + tid] = ldg_a_reg[i];
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            bs_vec[BLOCK_THREADS * i + tid] = ldg_b_reg[i];
        }
    }

    __device__ __forceinline__ void load_matrix(scalar_t *as, scalar_t *bs) {
        int warp_y = wid / BLOCK_N_WARPS * WARP_M;
        int warp_x = wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
            int warp_atom_offset_y = warp_y + i * WARP_ATOM_M;
#pragma unroll
            for (int k = 0; k < WARP_K_STEPS; ++k) {
                wmma.load_matrix_a(fa[k][i], as + warp_atom_offset_y * BLOCK_K + k * WARP_ATOM_K, BLOCK_K);
            }
        }
#pragma unroll
        for (int j = 0; j < WARP_N_STEPS; ++j) {
            int warp_atom_offset_x = warp_x + j * WARP_ATOM_N;
#pragma unroll
            for (int k = 0; k < WARP_K_STEPS; ++k) {
                wmma.load_matrix_b(fa[k][j], bs + warp_atom_offset_x + k * BLOCK_N, BLOCK_N);
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
                    nvcuda::wmma::mma_sync(fo[i][j], fa[k][i], fb[k][j], fo[i][j]);
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

} // namespace hgemm
