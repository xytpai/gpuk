#include "ops.h"
#include "rope_rms_impl.h"

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

template <typename T>
struct KernelElementType {
    using type = T;
};

template <>
struct KernelElementType<c10::Half> {
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16> {
    using type = hip_bfloat16;
};

void fused_rope_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, double eps) {
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() && cos_sin.is_contiguous());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides = positions.strides();
    TORCH_CHECK(pos_strides.size() == 1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv.scalar_type(),
        "fused_rope_rms", [&] {
            rope_rms::fused_rope_rms<scalar_t>(
                qkv.data_ptr<scalar_t>(),
                qw.data_ptr<scalar_t>(),
                kw.data_ptr<scalar_t>(),
                cos_sin.data_ptr<scalar_t>(),
                positions.data_ptr<int64_t>(),
                0,
                pos_strides[0],
                num_tokens,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                stream);
        });
}

void fused_mrope_3d_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                        int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                        bool is_neox_style, std::vector<int64_t> mrope_section_, bool is_interleaved, double eps) {
    TORCH_CHECK(mrope_section_.size() == 3);
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() && cos_sin.is_contiguous());
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides = positions.strides();
    TORCH_CHECK(pos_strides.size() == 2);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv.scalar_type(),
        "fused_mrope_3d_rms", [&] {
            using T = KernelElementType<scalar_t>::type;
            rope_rms::fused_mrope_rms<T, 3>(
                (T *)qkv.data_ptr<scalar_t>(),
                (T *)qw.data_ptr<scalar_t>(),
                (T *)kw.data_ptr<scalar_t>(),
                (T *)cos_sin.data_ptr<scalar_t>(),
                positions.data_ptr<int64_t>(),
                pos_strides[0],
                pos_strides[1],
                num_tokens,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                mrope_section,
                is_interleaved,
                stream);
        });
}

void fused_mrope_3d_rms_set_kv(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                               int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                               bool is_neox_style, std::vector<int64_t> mrope_section_, bool is_interleaved, double eps,
                               Tensor &q, Tensor &k_cache, Tensor &v_cache, Tensor &kv_loc, double k_scale, double v_scale) {
    TORCH_CHECK(mrope_section_.size() == 3);
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() && cos_sin.is_contiguous());
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && kv_loc.is_contiguous());
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides = positions.strides();
    auto kv_cache_dtype = k_cache.scalar_type();
    auto qkv_dtype = qkv.scalar_type();
    TORCH_CHECK(pos_strides.size() == 2);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv_dtype,
        "fused_mrope_3d_rms_set_kv", [&] {
            using T = KernelElementType<scalar_t>::type;
            if (kv_cache_dtype == qkv_dtype) {
                rope_rms::fused_mrope_rms_set_kv<T, 3, T>(
                    (T *)qkv.data_ptr<scalar_t>(),
                    (T *)qw.data_ptr<scalar_t>(),
                    (T *)kw.data_ptr<scalar_t>(),
                    (T *)cos_sin.data_ptr<scalar_t>(),
                    positions.data_ptr<int64_t>(),
                    pos_strides[0],
                    pos_strides[1],
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    mrope_section,
                    is_interleaved,
                    (T *)q.data_ptr<scalar_t>(),
                    (T *)k_cache.data_ptr<scalar_t>(),
                    (T *)v_cache.data_ptr<scalar_t>(),
                    kv_loc.data_ptr<int64_t>(),
                    (float)k_scale,
                    (float)v_scale,
                    stream);
            } else {
                rope_rms::fused_mrope_rms_set_kv<T, 3, fp8e4m3fn>(
                    (T *)qkv.data_ptr<scalar_t>(),
                    (T *)qw.data_ptr<scalar_t>(),
                    (T *)kw.data_ptr<scalar_t>(),
                    (T *)cos_sin.data_ptr<scalar_t>(),
                    positions.data_ptr<int64_t>(),
                    pos_strides[0],
                    pos_strides[1],
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    mrope_section,
                    is_interleaved,
                    (T *)q.data_ptr<scalar_t>(),
                    (fp8e4m3fn *)k_cache.data_ptr(),
                    (fp8e4m3fn *)v_cache.data_ptr(),
                    kv_loc.data_ptr<int64_t>(),
                    (float)k_scale,
                    (float)v_scale,
                    stream);
            }
        });
}
