#pragma once

#include "device_common.h"

template <typename T>
struct neg_zero {
    static constexpr T value = -T(0);
};

template <>
struct neg_zero<__half> {
    static constexpr unsigned short neg_zero_bits = 0x8000U;
    static constexpr __half value = __half_raw{neg_zero_bits};
    using bits_type = unsigned short;
};

template <>
struct neg_zero<__bfloat16> {
    static constexpr unsigned short neg_zero_bits = 0x8000U;
    static constexpr __bfloat16 value = __bfloat16_raw{neg_zero_bits};
    using bits_type = unsigned short;
};

template <>
struct neg_zero<float> {
    static constexpr unsigned int neg_zero_bits = 0x80000000U;
    static constexpr float value = -0.0f;
    using bits_type = unsigned int;
};

template <>
struct neg_zero<double> {
    static constexpr uint64_t neg_zero_bits = 0x8000000000000000ULL;
    static constexpr double value = -0.0f;
    using bits_type = uint64_t;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
    return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
    return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
    return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
    return (__half_as_ushort(x) == 0x8000);
}

// __bfloat16 specialization
template <>
__device__ bool is_negative_zero<__bfloat16>(__bfloat16 x) {
    return (__bfloat16_as_ushort(x) == 0x8000);
}
