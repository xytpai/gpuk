#pragma once

#include "device_common.h"

template <typename T>
struct neg_zero {
};

template <>
struct neg_zero<__half> {
    using bits_type = unsigned short;
    static constexpr unsigned short neg_zero_bits = 0x8000U;
};

template <>
struct neg_zero<__bfloat16> {
    using bits_type = unsigned short;
    static constexpr unsigned short neg_zero_bits = 0x8000U;
};

template <>
struct neg_zero<float> {
    using bits_type = unsigned int;
    static constexpr unsigned int neg_zero_bits = 0x80000000U;
};

template <>
struct neg_zero<double> {
    using bits_type = uint64_t;
    static constexpr uint64_t neg_zero_bits = 0x8000000000000000ULL;
};

// template <typename T>
// __device__ __forceinline__ bool is_negative_zero(T) {
//     return false;
// }

// // float specialization
// template <>
// __device__ __forceinline__ bool is_negative_zero<float>(float x) {
//     return (__float_as_int(x) == 0x80000000);
// }

// // double specialization
// template <>
// __device__ __forceinline__ bool is_negative_zero<double>(double x) {
//     return (__double_as_longlong(x) == 0x8000000000000000ULL);
// }

// // __half specialization
// template <>
// __device__ __forceinline__ bool is_negative_zero<__half>(__half x) {
//     return (__half_as_ushort(x) == 0x8000);
// }

// // __bfloat16 specialization
// template <>
// __device__ bool is_negative_zero<__bfloat16>(__bfloat16 x) {
//     return (__bfloat16_as_ushort(x) == 0x8000);
// }
