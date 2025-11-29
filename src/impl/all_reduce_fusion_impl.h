#pragma once

#include "device_common.h"

using namespace std;
using namespace kernel_utils;
namespace cg = cooperative_groups;

#define WARP_SIZE 64
#define NBLOCKS_PER_GPU 256

namespace details {

static constexpr int kBytesPerAccess = 16;

__device__ __forceinline__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
    __scoped_atomic_store_n(addr, flag, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
#endif
}

__device__ __forceinline__ int ld_flag(int *addr) {
    int flag;
#ifdef __CUDACC__
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
#else
    flag =
        __scoped_atomic_load_n(addr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
#endif
    return flag;
}

} // namespace details

template <int NRanks>
struct CommPtrs {
    std::array<void *, NRanks> barrier_flag_ptrs;
    std::array<void *, NRanks> data_ptrs;
    void *sync_clock;
    int rank;
};

struct HostCommPtrs {
    std::vector<void *> barrier_flag_ptrs;
    std::vector<void *> data_ptrs;
    void *sync_clock;
    int rank;
    int nranks;
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(CommPtrs<NRanks> &cptrs) {
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = cptrs.data_ptrs[r];
            barrier_flags[r] = cptrs.barrier_flag_ptrs[r];
        }
        flag_ptr = ((int *)cptrs.sync_clock) + blockIdx.x;
        int rank = cptrs.rank;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            int target_rank = threadIdx.x;
            target_flag = reinterpret_cast<int *>(barrier_flags[target_rank]) + blockIdx.x * NRanks + rank;
            current_flag = reinterpret_cast<int *>(barrier_flags[rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        auto flag = (*flag_ptr) + 1;
        if (threadIdx.x < NRanks) {
            details::st_flag(target_flag, flag);
            while (details::ld_flag(current_flag) < flag) {
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *flag_ptr = flag;
        }
    }

    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int *target_flag;
    int *current_flag;
};

enum QuantType {
    NONE = 0,
    FP8E4M3FN = 1,
    FP8E4M3FNUZ = 2,
};

template <typename T>
struct AllReduceFusionParams {
    int nranks;
    int rank;
    int size;
    int hidden_dim;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    float rms_eps;
    // per token quant
    QuantType quant_type;
    void *scale_out;
};

template <typename T, int VEC_SIZE, typename QuantT>
__device__ __forceinline__ vec_t<QuantT, VEC_SIZE> convert_to_fp8(vec_t<T, VEC_SIZE> &in_vec, float scale) {
    vec_t<QuantT, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        volatile float out = static_cast<float>(in_vec[i]) / scale;
        out_vec[i] = static_cast<QuantT>(out);
    }
    return out_vec;
}

template <typename T, int VEC_SIZE, typename OutT>
__device__ __forceinline__ vec_t<OutT, VEC_SIZE> rms_norm(AllReduceFusionParams<T> const &m_params,
                                                          vec_t<T, VEC_SIZE> const &residual, vec_t<T, VEC_SIZE> const &gamma) {
    __shared__ float s_val;
    vec_t<OutT, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]);
        acc += v * v;
    }
    acc = block_reduce<float, WARP_SIZE>(acc, std::plus<float>());
    if (threadIdx.x == 0) {
        s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float out = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]) * s_val * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]);
        norm_out[i] = static_cast<OutT>(out);
    }
    return norm_out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ float reduce_abs_max(vec_t<T, VEC_SIZE> const &data) {
    __shared__ float s_val;
    auto fn = [](float a, float b) { return a > b ? a : b; };
    float acc = -1.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&data)[i]);
        acc = fn(acc, std::abs(v));
    }
    acc = block_reduce<float, WARP_SIZE>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE, bool STORE = true>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
    if constexpr (STORE)
        rms_in.store(reinterpret_cast<T *>(params.residual_out) + idx);
    if (params.quant_type == QuantType::NONE) {
        auto val = rms_norm<T, VEC_SIZE, T>(params, rms_in, rms_weight);
        val.store(reinterpret_cast<T *>(params.norm_out) + idx);
    } else {
        auto val = rms_norm<T, VEC_SIZE, float>(params, rms_in, rms_weight);
        float scale = reduce_abs_max<float, VEC_SIZE>(val);
        if (params.quant_type == QuantType::FP8E4M3FN) {
            scale = scale == 0.f ? 1.f : scale / fp8e4m3fn::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fn>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fn *>(params.norm_out) + idx);
        } else {
            scale = scale == 0.f ? 1.f : scale / fp8e4m3fnuz::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fnuz>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fnuz *>(params.norm_out) + idx);
        }
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(params.scale_out)[tidx] = scale;
    }
}

template <typename T, int NRanks, bool FETCH>
__global__ void allreduce_fusion_kernel_twoshot_direct(AllReduceFusionParams<T> params, CommPtrs<NRanks> cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id_begin =
        (blockIdx.x * NRanks + 0) * params.hidden_dim + access_id_in_token;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    SyncComm<NRanks> comm(cptrs);

    if constexpr (FETCH) {
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            for (int idx =
                     (blockIdx.x * NRanks + r) * params.hidden_dim + access_id_in_token;
                 idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
                reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
                    reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];
            }
        }
    }

    comm.sync();

    // allreduce
    for (int idx = (blockIdx.x * NRanks + params.rank) * params.hidden_dim + access_id_in_token;
         idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
        vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx);
        }
        vec_add_r_<T, VEC_SIZE, NRanks>(vals);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[0].store(reinterpret_cast<T *>(comm.comm_bufs[r]) + params.size + idx);
        }
    }

    comm.sync();

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        int token_id = blockIdx.x * NRanks + r;
        for (int idx = token_id * params.hidden_dim + access_id_in_token, tidx = token_id;
             idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim, tidx += gridDim.x * NRanks) {
            vec_t<T, VEC_SIZE> data[2];
            data[0].load(reinterpret_cast<T *>(params.residual_in) + idx);
            data[1].load(reinterpret_cast<T *>(comm.comm_bufs[params.rank]) + params.size + idx);
            vec_add_<T, VEC_SIZE>(data[0], data[1]);
            epilogue<T, VEC_SIZE>(params, data[0], gamma, idx, tidx);
        }
    }
}

template <typename T, int NRanks, bool FETCH>
__global__ void allreduce_fusion_kernel_single_load(AllReduceFusionParams<T> params, CommPtrs<NRanks> cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int tidx = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int idx = tidx * params.hidden_dim + access_id_in_token;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
    SyncComm<NRanks> comm(cptrs);

    if constexpr (FETCH) {
        reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
            reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];
    }

    comm.sync();

    // cross-device load
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        auto r_ = (params.rank + r) % NRanks;
        vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r_]) + idx);
    }
    vec_add_r_<T, VEC_SIZE, NRanks>(vals);

    vec_t<T, VEC_SIZE> residual;
    residual.load(reinterpret_cast<T *>(params.residual_in) + idx);
    vec_add_<T, VEC_SIZE>(residual, vals[0]);
    epilogue<T, VEC_SIZE>(params, residual, gamma, idx, tidx);
}

// template <typename T, int NRanks>
// __global__ void allreduce_kernel(AllReduceFusionParams<T> params) {
//     static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
//     SyncComm<NRanks> comm(params.workspace, params.rank);
//     int warp_id = threadIdx.x / WARP_SIZE;
//     int lane_id = threadIdx.x % WARP_SIZE;
//     __shared__ T shared[NRanks * WARP_SIZE * VEC_SIZE];
//     int part = params.size / (VEC_SIZE * WARP_SIZE) / NRanks;
//     comm.sync();
//     for (int bid = blockIdx.x; bid < part; bid += gridDim.x) {
//         int idx = ((params.rank * part + bid) * WARP_SIZE + lane_id) * VEC_SIZE;
//         vec_t<T, VEC_SIZE> val;
//         val.load(reinterpret_cast<T *>(comm.comm_bufs[warp_id]) + idx);
//         val.store(&shared[0] + threadIdx.x * VEC_SIZE);
//         __syncthreads();
//         if (warp_id == 0) {
//             vec_t<T, VEC_SIZE> acc, vec;
//             acc.load(&shared[0] + lane_id * VEC_SIZE);
// #pragma unroll
//             for (int r = 1; r < NRanks; ++r) {
//                 vec.load(&shared[0] + (r * WARP_SIZE + lane_id) * VEC_SIZE);
//                 vec_add_<T, VEC_SIZE>(acc, vec);
//             }
//             acc.store(&shared[0] + lane_id * VEC_SIZE);
//         }
//         __syncthreads();
//         val.load(&shared[0] + lane_id * VEC_SIZE);
//         vec_t<T, VEC_SIZE> res;
//         res.load(reinterpret_cast<T *>(params.residual_in) + idx);
//         vec_add_<T, VEC_SIZE>(val, res);
//         val.store(reinterpret_cast<T *>(comm.comm_bufs[warp_id]) + params.size + idx);
//     }
//     comm.sync();
// }

// template <typename T, int NRanks, int LOOPS>
// __global__ void rms_kernel(AllReduceFusionParams<T> params) {
//     static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
//     int tidx = blockIdx.x;
//     int access_id_in_token = threadIdx.x * VEC_SIZE;
//     vec_t<T, VEC_SIZE> gamma;
//     gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
// #pragma unroll
//     for (int i = 0; i < LOOPS; ++i) {
//         int idx = tidx * params.hidden_dim + access_id_in_token;
//         if (idx < params.size) {
//             vec_t<T, VEC_SIZE> val;
//             val.load(reinterpret_cast<T *>(params.workspace[params.rank]) + params.size + idx);
//             epilogue<T, VEC_SIZE, true>(params, val, gamma, idx, tidx);
//         }
//         tidx += gridDim.x;
//     }
// }

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      CommPtrs<NRanks> const &cptrs,
                                      gpuStream_t stream) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    dim3 threadsPerBlock(threads_per_token);
    // void *args[] = {(void *)&params};
    if (token_num <= 1024) {
        dim3 numBlocks(token_num);
        allreduce_fusion_kernel_single_load<T, NRanks, false><<<numBlocks, threadsPerBlock, 0, stream>>>(params, cptrs);
    } else {
        int nblocks = std::min((token_num + NRanks - 1) / NRanks, NBLOCKS_PER_GPU);
        if (params.size * sizeof(T) >= 1024 * 1024 * 128) {
            nblocks /= 2;
        }
        dim3 numBlocks(nblocks);
        allreduce_fusion_kernel_twoshot_direct<T, NRanks, false><<<numBlocks,
                                                                   threadsPerBlock, 0, stream>>>(params, cptrs);
    }
}

template <typename T>
void allreduce_rms_fusion_impl(HostCommPtrs host_cptrs, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = host_cptrs.nranks;
    params.rank = host_cptrs.rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.allreduce_in = allreduce_in;
    params.residual_in = residual_in;
    params.residual_out = residual_out;
    params.norm_out = norm_out;
    params.rms_gamma = rms_gamma;
    params.rms_eps = eps;
    params.scale_out = scale_out;
    params.quant_type = (QuantType)quant_type;

#define DISPATCH_NRANKS(NRANKS)                                             \
    {                                                                       \
        CommPtrs<NRANKS> cptrs;                                             \
        for (int i = 0; i < NRANKS; ++i) {                                  \
            cptrs.barrier_flag_ptrs[i] = host_cptrs.barrier_flag_ptrs[i];   \
            cptrs.data_ptrs[i] = host_cptrs.data_ptrs[i];                   \
        }                                                                   \
        cptrs.sync_clock = host_cptrs.sync_clock;                           \
        cptrs.rank = host_cptrs.rank;                                       \
        allreduce_fusion_kernel_launcher<T, NRANKS>(params, cptrs, stream); \
    }

    int nranks = host_cptrs.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        assert(false);
    }

#undef DISPATCH_NRANKS
}
