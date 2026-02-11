#pragma once

#include "device_common.h"

using namespace std;
using namespace kernel_utils;
using namespace comm_utils;
// namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define CHECK_FAIL assert

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
    // retain
    void *allreduce_out;
};

// ========================================= allreduce =========================================

template <typename T, int NRanks, int BLOCK_SIZE, int BYTES_PER_ACCESS>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_inplace_kernel_2stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *cptrs) {
    constexpr int VEC_SIZE = BYTES_PER_ACCESS / sizeof(T);
    constexpr int BLOCK_WORK_SIZE = BLOCK_SIZE * VEC_SIZE;
    constexpr int WARP_SIZE_ = BLOCK_SIZE / NRanks;
    SyncComm<NRanks> comm(meta);
    __shared__ T shared[NRanks * WARP_SIZE_ * VEC_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE_;
    int lane_id = threadIdx.x % WARP_SIZE_;
    vec_t<T, VEC_SIZE> val;
    vec_t<float, VEC_SIZE> acc;
    for (
        int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * VEC_SIZE;
        idx < params.size;
        idx += gridDim.x * BLOCK_SIZE * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(params.allreduce_in) + idx);
        val.store(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
    }
    comm.template sync<false, false>();
    for (
        int idx = ((blockIdx.x * NRanks + params.rank) * WARP_SIZE_ + lane_id) * VEC_SIZE;
        idx < params.size;
        idx += gridDim.x * NRanks * WARP_SIZE_ * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
        val.store(&shared[0] + threadIdx.x * VEC_SIZE);
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc.data[v] = (float)val.data[v];
            }
#pragma unroll
            for (int r = 1; r < NRanks; ++r) {
                val.load(&shared[0] + (r * WARP_SIZE_ + lane_id) * VEC_SIZE);
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    acc.data[v] += (float)val.data[v];
                }
            }
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                val.data[v] = (T)acc.data[v];
            }
            val.store(&shared[0] + lane_id * VEC_SIZE);
        }
        __syncthreads();
        val.load(&shared[0] + lane_id * VEC_SIZE);
        val.store(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
    }
    comm.template sync<false, true>();
    for (
        int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * VEC_SIZE;
        idx < params.size;
        idx += gridDim.x * BLOCK_SIZE * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
        val.store(reinterpret_cast<T *>(params.allreduce_in) + idx);
    }
}

template <typename T, int NRanks, int BLOCK_SIZE = 512, int BYTES_PER_ACCESS = 16>
void allreduce_inplace_kernel_2stage_launcher(
    AllReduceFusionParams<T> const &params,
    CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = BYTES_PER_ACCESS / sizeof(T);
    constexpr int BLOCK_WORK_SIZE = BLOCK_SIZE * VEC_SIZE;
    dim3 threadsPerBlock(BLOCK_SIZE);
    int nblocks = (params.size + BLOCK_WORK_SIZE - 1) / BLOCK_WORK_SIZE;
    nblocks = std::min(nblocks, NBLOCKS_PER_GPU);
    dim3 numBlocks(nblocks);
    allreduce_inplace_kernel_2stage<T, NRanks, BLOCK_SIZE, BYTES_PER_ACCESS><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

template <typename T>
void allreduce_inplace_impl(CommMeta meta, CommPtrs *cptrs, void *allreduce_in, int size, gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = meta.nranks;
    params.rank = meta.rank;
    params.size = size;
    params.allreduce_in = allreduce_in;
#define DISPATCH_NRANKS(NRANKS)                                                            \
    {                                                                                      \
        CommDeviceMeta<NRANKS> dmeta;                                                      \
        for (int i = 0; i < NRANKS; ++i) {                                                 \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                        \
        }                                                                                  \
        dmeta.sync_clock = meta.sync_clock;                                                \
        dmeta.rank = meta.rank;                                                            \
        dmeta.nranks = meta.nranks;                                                        \
        allreduce_inplace_kernel_2stage_launcher<T, NRANKS>(params, dmeta, cptrs, stream); \
    }
    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        CHECK_FAIL(false);
    }
#undef DISPATCH_NRANKS
}

// ========================================= allreduce fusion =========================================

template <typename T, int VEC_SIZE, typename QuantT>
__device__ __forceinline__ vec_t<QuantT, VEC_SIZE> convert_to_fp8(vec_t<T, VEC_SIZE> &in_vec, float scale) {
    vec_t<QuantT, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float out = static_cast<float>(in_vec[i]) / scale;
        out_vec[i] = static_cast<QuantT>(out);
    }
    return out_vec;
}

template <typename T, int VEC_SIZE, typename OutT, int BLOCK_SIZE>
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
    acc = block_reduce<float, WARP_SIZE, BLOCK_SIZE>(acc, std::plus<float>());
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

template <typename T, int VEC_SIZE, int BLOCK_SIZE>
__device__ __forceinline__ float reduce_abs_max(vec_t<T, VEC_SIZE> const &data) {
    __shared__ float s_val;
    auto fn = [](float a, float b) { return a > b ? a : b; };
    float acc = -1.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&data)[i]);
        acc = fn(acc, std::abs(v));
    }
    acc = block_reduce<float, WARP_SIZE, BLOCK_SIZE>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE, bool STORE = true, int BLOCK_SIZE = 0, int QUANT_TYPE = 0>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
    if constexpr (STORE)
        rms_in.store(reinterpret_cast<T *>(params.residual_out) + idx);
    if constexpr (QUANT_TYPE == QuantType::NONE) {
        auto val = rms_norm<T, VEC_SIZE, T, BLOCK_SIZE>(params, rms_in, rms_weight);
        val.store(reinterpret_cast<T *>(params.norm_out) + idx);
    } else {
        auto val = rms_norm<T, VEC_SIZE, float, BLOCK_SIZE>(params, rms_in, rms_weight);
        float scale = reduce_abs_max<float, VEC_SIZE, BLOCK_SIZE>(val);
        if constexpr (QUANT_TYPE == QuantType::FP8E4M3FN) {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fn::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fn>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fn *>(params.norm_out) + idx);
        } else {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fnuz::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fnuz>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fnuz *>(params.norm_out) + idx);
        }
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(params.scale_out)[tidx] = scale;
    }
}

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_1stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *__restrict__ cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    SyncComm<NRanks> comm(meta);
    comm.sync();
    using vec_t_ = vec_t<T, VEC_SIZE>;
    using acc_vec_t_ = vec_t<float, VEC_SIZE>;
    int tidx = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int idx = tidx * params.hidden_dim + access_id_in_token;

    acc_vec_t_ acc;
    auto vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        acc.data[v] = vec.data[v];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
        vec = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(cptrs->data_ptrs[r]) + idx);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            acc.data[v] += (float)vec.data[v];
        }
    }
    auto res = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.residual_in) + idx);
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        acc.data[v] += (float)res.data[v];
    }
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
        vec.data[v] = (T)acc.data[v];
    }
    *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.residual_out) + idx) = vec;
    auto gamma = *reinterpret_cast<vec_t_ *>(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
    epilogue<T, VEC_SIZE, false, BLOCK_SIZE, QUANT_TYPE>(params, vec, gamma, idx, tidx);
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_1stage_launcher(
    AllReduceFusionParams<T> const &params,
    CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    CHECK_FAIL(token_num <= NBLOCKS_PER_GPU);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_1stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

template <typename T, int NRanks, int BLOCK_SIZE, int QUANT_TYPE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_2stage(
    AllReduceFusionParams<T> params, CommDeviceMeta<NRanks> meta, CommPtrs *cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int WARP_SIZE_ = BLOCK_SIZE / NRanks;
    SyncComm<NRanks> comm(meta);

    __shared__ T shared[NRanks * WARP_SIZE_ * VEC_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE_;
    int lane_id = threadIdx.x % WARP_SIZE_;

    vec_t<T, VEC_SIZE> val;
    vec_t<float, VEC_SIZE> acc;

    comm.template sync<true, false>();

    for (
        int idx = ((blockIdx.x * NRanks + params.rank) * WARP_SIZE_ + lane_id) * VEC_SIZE;
        idx < params.size;
        idx += gridDim.x * NRanks * WARP_SIZE_ * VEC_SIZE) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
        val.store(&shared[0] + threadIdx.x * VEC_SIZE);
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                acc.data[v] = (float)val.data[v];
            }
#pragma unroll
            for (int r = 1; r < NRanks; ++r) {
                val.load(&shared[0] + (r * WARP_SIZE_ + lane_id) * VEC_SIZE);
#pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    acc.data[v] += (float)val.data[v];
                }
            }
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
                val.data[v] = (T)acc.data[v];
            }
            val.store(&shared[0] + lane_id * VEC_SIZE);
        }
        __syncthreads();
        val.load(&shared[0] + lane_id * VEC_SIZE);
        val.store(reinterpret_cast<T *>(cptrs->data_ptrs[warp_id]) + idx);
    }

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
    comm.template sync<false, true>();
    for (
        int idx = blockIdx.x * params.hidden_dim + access_id_in_token, tidx = blockIdx.x;
        idx < params.size;
        idx += gridDim.x * params.hidden_dim, tidx += gridDim.x) {
        val.load(reinterpret_cast<T *>(cptrs->data_ptrs[0]) + idx);
        vec_t<T, VEC_SIZE> res;
        res.load(reinterpret_cast<T *>(params.residual_in) + idx);
        val.add_(res);
        epilogue<T, VEC_SIZE, true, BLOCK_SIZE, QUANT_TYPE>(params, val, gamma, idx, tidx);
    }
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_2stage_launcher(
    AllReduceFusionParams<T> const &params,
    CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    int token_num = params.size / params.hidden_dim;
    dim3 threadsPerBlock(BLOCK_SIZE);
    token_num = std::min(token_num, NBLOCKS_PER_GPU);
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_2stage<T, NRanks, BLOCK_SIZE, QUANT_TYPE><<<numBlocks, threadsPerBlock, 0, stream>>>(params, meta, cptrs);
}

template <typename T, int NRanks, int HIDDEN_DIM, int QUANT_TYPE>
void allreduce_fusion_kernel_launcher_(
    AllReduceFusionParams<T> const &params,
    CommDeviceMeta<NRanks> const &meta,
    CommPtrs *cptrs,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_num = params.size / params.hidden_dim;
    CHECK_FAIL(params.size % params.hidden_dim == 0);
    CHECK_FAIL(params.hidden_dim % VEC_SIZE == 0);
    CHECK_FAIL(params.hidden_dim == HIDDEN_DIM);
    auto bytes = params.size * sizeof(T);
    bool use_1s = token_num <= (NBLOCKS_PER_GPU / 4);
    use_1s = use_1s && ((NRanks <= 2) || (NRanks <= 4 && bytes < 160 * 1024) || (NRanks <= 8 && bytes < 80 * 1024));
    if (use_1s) {
        allreduce_fusion_kernel_1stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    } else {
        allreduce_fusion_kernel_2stage_launcher<T, NRanks, HIDDEN_DIM, QUANT_TYPE>(params, meta, cptrs, stream);
    }
}

template <typename T, int NRanks, int QUANT_TYPE>
void allreduce_fusion_kernel_launcher_hd(AllReduceFusionParams<T> const &params,
                                         CommDeviceMeta<NRanks> const &meta,
                                         CommPtrs *cptrs,
                                         gpuStream_t stream) {
    switch (params.hidden_dim) {
    case 4096:
        allreduce_fusion_kernel_launcher_<T, NRanks, 4096, QUANT_TYPE>(params, meta, cptrs, stream);
        return;
    case 2048:
        allreduce_fusion_kernel_launcher_<T, NRanks, 2048, QUANT_TYPE>(params, meta, cptrs, stream);
        return;
    case 1024:
        allreduce_fusion_kernel_launcher_<T, NRanks, 1024, QUANT_TYPE>(params, meta, cptrs, stream);
        return;
    default:
        CHECK_FAIL(false);
    }
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      CommDeviceMeta<NRanks> const &meta,
                                      CommPtrs *cptrs,
                                      gpuStream_t stream) {
    switch (params.quant_type) {
    case QuantType::NONE:
        allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::NONE>(params, meta, cptrs, stream);
        return;
    case QuantType::FP8E4M3FN:
        allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FN>(params, meta, cptrs, stream);
        return;
    case QuantType::FP8E4M3FNUZ:
        allreduce_fusion_kernel_launcher_hd<T, NRanks, QuantType::FP8E4M3FNUZ>(params, meta, cptrs, stream);
        return;
    default:
        CHECK_FAIL(false);
    }
}

template <typename T>
void allreduce_rms_fusion_impl(CommMeta meta, CommPtrs *cptrs, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = meta.nranks;
    params.rank = meta.rank;
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

#define DISPATCH_NRANKS(NRANKS)                                                    \
    {                                                                              \
        CommDeviceMeta<NRANKS> dmeta;                                              \
        for (int i = 0; i < NRANKS; ++i) {                                         \
            dmeta.barrier_flag_ptrs[i] = meta.barrier_flag_ptrs[i];                \
        }                                                                          \
        dmeta.sync_clock = meta.sync_clock;                                        \
        dmeta.rank = meta.rank;                                                    \
        dmeta.nranks = meta.nranks;                                                \
        allreduce_fusion_kernel_launcher<T, NRANKS>(params, dmeta, cptrs, stream); \
    }

    int nranks = meta.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        CHECK_FAIL(false);
    }

#undef DISPATCH_NRANKS
}
