#pragma once

#include "device_common.h"

using namespace std;
using namespace kernel_utils;
namespace cg = cooperative_groups;

#define NBLOCKS_PER_GPU 256

namespace details {

static constexpr int kBytesPerAccess = 16;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kOneShotMaxSize =
    kOneShotMaxToken * 1024 * kBytesPerAccess;

} // namespace details

namespace comm {

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
class Barrier {
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const &comm) {
        if (threadIdx.x < NRanks) {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int *>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag =
                reinterpret_cast<int *>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        constexpr int kBarrierFlagCount = NBLOCKS_PER_GPU;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            m_flag_value = next_flag(m_flag_value);
            // To avoid the ABA problem, we need to synchronize the correct flag value
            // to all barrier_flags, even if the corresponding CTA has not been
            // launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount;
                 flag_idx += gridDim.x) {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
            }
        }
        __syncthreads();
    }

protected:
    __device__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        __scoped_atomic_store_n(addr, flag, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
#endif
    }

    __device__ int ld_flag(int *addr) {
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

    __device__ __forceinline__ int next_flag(int flag) {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag) {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    volatile int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};

template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void **workspace, int rank) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 2];
        int comm_size = *reinterpret_cast<int *>(workspace[NRanks * 3 + 3]);
        clear_ptr = (int *)workspace[NRanks * 3 + 4];
        flag_value = *flag_ptr;
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + r]) + static_cast<int64_t>(data_offset) * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    int *clear_ptr;
    uint8_t *data_bufs[NRanks];
    uint8_t *clear_buf;
    int clear_size;
    int flag_value;
};

} // namespace comm

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
    void **workspace;
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
    acc = block_reduce<float>(acc, std::plus<float>());
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
    acc = block_reduce<float>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
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

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_direct(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id_begin =
        (blockIdx.x * NRanks + 0) * params.hidden_dim + access_id_in_token;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::SyncComm<NRanks> comm(params.workspace);

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        for (int idx =
                 (blockIdx.x * NRanks + r) * params.hidden_dim + access_id_in_token;
             idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
            reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
                reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];
        }
    }

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

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

    barrier.sync();

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

    comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_id = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id = token_id * params.hidden_dim + access_id_in_token;
    int access_stride = gridDim.x * params.hidden_dim;

    vec_t<T, VEC_SIZE> clear_vec;
    clear_vec.cast_fill(neg_zero<T>::neg_zero_bits);

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size;

    for (int idx = access_id; idx < params.size; idx += access_stride) {
        vec_t<T, VEC_SIZE> val;
        val.load(reinterpret_cast<T *>(params.allreduce_in) + idx);
        remove_neg_zero<T, VEC_SIZE>(val);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            // Push data to other ranks
            val.store(reinterpret_cast<T *>(comm.data_bufs[r]) + params.rank * params.size + idx);
        }
    }

    for (int idx = access_id; idx < clear_access; idx += access_stride) {
        // Clear comm buffer that previous kernel used
        clear_vec.store(reinterpret_cast<T *>(comm.clear_buf) + idx);
    }

    for (int idx = access_id, tidx = token_id; idx < params.size;
         idx += access_stride, tidx += gridDim.x) {
        vec_t<T, VEC_SIZE> residual;
        residual.load(reinterpret_cast<T *>(params.residual_in) + idx);

        vec_t<T, VEC_SIZE> vals[NRanks];
        volatile bool done = false;
        while (!done) {
            done = true;
            __threadfence();
#pragma unroll
            for (int r = 0; r < NRanks; ++r) {
                // LDG.128 from local rank
                vals[r].load(reinterpret_cast<T *>(comm.data_bufs[params.rank]) + r * params.size + idx);
                done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
            }
        }
        vec_add_r_<T, VEC_SIZE, NRanks>(vals);
        vec_add_<T, VEC_SIZE>(vals[0], residual);
        epilogue<T, VEC_SIZE>(params, vals[0], gamma, idx, tidx);
    }

    comm.update(params.size * NRanks);
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_single_load(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::SyncComm<NRanks> comm(params.workspace);

    int idx = blockIdx.x * params.hidden_dim + access_id_in_token;
    reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
        reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

    // cross-device load
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx);
    }
    vec_add_r_<T, VEC_SIZE, NRanks>(vals);

    int tidx = blockIdx.x;
    vec_t<T, VEC_SIZE> residual;
    residual.load(reinterpret_cast<T *>(params.residual_in) + idx);
    vec_add_<T, VEC_SIZE>(residual, vals[0]);
    epilogue<T, VEC_SIZE>(params, residual, gamma, idx, tidx);

    comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      gpuStream_t stream) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    dim3 threadsPerBlock(threads_per_token);
    // void *args[] = {(void *)&params};
    if (token_num <= NBLOCKS_PER_GPU) {
        dim3 numBlocks(token_num);
        allreduce_fusion_kernel_twoshot_single_load<T, NRanks><<<numBlocks,
                                                                 threadsPerBlock, 0, stream>>>(params);
    } else {
        int nblocks = std::min(token_num, NBLOCKS_PER_GPU);
        if (params.size * sizeof(T) >= 1024 * 1024 * 128) {
            nblocks /= 2;
        }
        dim3 numBlocks(nblocks);
        allreduce_fusion_kernel_twoshot_direct<T, NRanks><<<numBlocks,
                                                            threadsPerBlock, 0, stream>>>(params);
    }
}

template <typename T>
void allreduce_rms_fusion_impl(void **workspace, int rank, int nranks, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = nranks;
    params.rank = rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.workspace = workspace;
    params.allreduce_in = allreduce_in;
    params.residual_in = residual_in;
    params.residual_out = residual_out;
    params.norm_out = norm_out;
    params.rms_gamma = rms_gamma;
    params.rms_eps = eps;
    params.scale_out = scale_out;
    params.quant_type = (QuantType)quant_type;
    if (nranks == 8) {
        allreduce_fusion_kernel_launcher<T, 8>(params, stream);
    } else if (nranks == 4) {
        allreduce_fusion_kernel_launcher<T, 4>(params, stream);
    } else if (nranks == 2) {
        allreduce_fusion_kernel_launcher<T, 2>(params, stream);
    } else {
        assert(false);
    }
}
