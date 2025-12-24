#pragma once

#include "device_common.h"

using namespace kernel_utils;

namespace attention {

template <typename T, int HEAD_SIZE, bool IS_CAUSAL = true>
__global__ void sdpa_ref_kernel(
    T *out, const T *q, const T *k, const T *v,
    int batch_size, int num_heads, int q_seq_length, int kv_seq_length,
    T *qk_temp, float neg_inf) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int q_seq_id = blockIdx.y;
    int offset_b_q = b * q_seq_length * HEAD_SIZE;
    int offset_b_kv = b * kv_seq_length * HEAD_SIZE;
    auto o_b = out + offset_b_q;
    auto q_b = q + offset_b_q;
    auto k_b = k + offset_b_kv;
    auto v_b = v + offset_b_kv;
    auto seq2 = qk_temp + b * q_seq_length * kv_seq_length;
    for (int kv_seq_id = 0; kv_seq_id < kv_seq_length; ++kv_seq_id) {
        auto q_s = q_b + q_seq_id * HEAD_SIZE;
        auto k_s = k_b + kv_seq_id * HEAD_SIZE;
        float sum = ((float)q_s[tid]) * ((float)k_s[tid]);
        sum = block_reduce<float, WARP_SIZE, BLOCK_SIZE>(acc, std::plus<float>());
        if (tid == 0) {
            seq2[q_seq_id * kv_seq_length + kv_seq_id] = sum * (1.0f / std::sqrt((float)HEAD_SIZE));
        }
    }
    __syncthreads();
    if constexpr (IS_CAUSAL) {
        for (int kv_seq_id = 0; kv_seq_id < kv_seq_length; ++kv_seq_id) {
            if (q_seq_id < kv_seq_id) {
                seq2[q_seq_id * kv_seq_length + kv_seq_id] = neg_inf;
            }
        }
        __syncthreads();
    }

    __shared__ float s_val;
    if (tid == 0) {
        float max_value = neg_inf;
        float e2sum = 0;
        for (int kv_seq_id = 0; kv_seq_id < kv_seq_length; ++kv_seq_id) {
            max_value = std::max(seq2[q_seq_id * kv_seq_length + kv_seq_id], max_value);
        }
        for (int kv_seq_id = 0; kv_seq_id < kv_seq_length; ++kv_seq_id) {
            e2sum += std::exp((float)seq2[q_seq_id * kv_seq_length + kv_seq_id] - max_value);
        }
        for (int kv_seq_id = 0; kv_seq_id < kv_seq_length; ++kv_seq_id) {
            seq2[q_seq_id * kv_seq_length + kv_seq_id] =
                std::exp(seq2[q_seq_id * kv_seq_length + kv_seq_id] - max_value) / e2sum;
        }
    }
    __syncthreads();
    for (int m = 0; m < q_seq_length_; m++) {
        for (int kk = 0; kk < hidden_size_; kk++) {
            float sum = 0;
            for (int n = 0; n < kv_seq_length_; n++) {
                sum += (float)(seq2[m * kv_seq_length_ + n] * v_b[n * hidden_size_ + kk]);
            }
            o_b[m * hidden_size_ + kk] = sum;
        }
    }
}

template <typename T, int HEAD_SIZE>
void launch_sdpa_ref(
    T *out, const T *q, const T *k, const T *v,
    int batch_size, int num_heads, int q_seq_length, int kv_seq_length,
    bool is_causal, T *qk_temp, gpuStream_t stream) {
    // q: batch_size, num_heads, q_seq_length, HEAD_SIZE
    // kv: batch_size, num_heads, kv_seq_length, HEAD_SIZE
    std::cout << "You're now entering the ref path, which will result in a significant performance drop.\n";
    float neg_inf = -std::numeric_limits<float>::infinity();
    dim3 threadsPerBlock(HEAD_SIZE);
    dim3 numBlocks(batch_size * num_heads, q_seq_length);
    if (is_causal) {
        sdpa_ref_kernel<T, HEAD_SIZE, true><<<numBlocks, threadsPerBlock, 0, stream>>>(
            out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, qk_temp, neg_inf);
    } else {
        sdpa_ref_kernel<T, HEAD_SIZE, false><<<numBlocks, threadsPerBlock, 0, stream>>>(
            out, q, k, v, batch_size, num_heads, q_seq_length, kv_seq_length, qk_temp, neg_inf);
    }
}

template <typename T, int HEAD_SIZE>
__global__ void sdpa_warp_kernel(
    T *out, const T *q, const T *k, const T *v,
    int batch_size, int num_heads_q, int num_heads_kv, int q_seq_length, int kv_seq_length) {
}

template <typename T, int HEAD_SIZE>
void sdpa_warp(
    T *out, const T *q, const T *k, const T *v,
    int batch_size, int num_heads_q, int num_heads_kv, int q_seq_length, int kv_seq_length) {
    // q: batch_size, q_seq_length, num_heads_q, HEAD_SIZE
    // kv: batch_size, kv_seq_length, num_heads_kv, HEAD_SIZE
    constexpr int WARP_SIZE = HEAD_SIZE;
}

} // namespace attention
