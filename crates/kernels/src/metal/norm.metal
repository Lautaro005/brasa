// RMSNorm por filas: out = x / sqrt(mean(x²) + eps) * w. Un threadgroup por fila.
// También sirve para QK-norm (filas = tokens × cabezas, n = head_dim).
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

kernel void rms_norm_f32(device const float* x   [[buffer(0)]],
                         device const float* w   [[buffer(1)]],
                         device float*       out [[buffer(2)]],
                         constant uint&      n   [[buffer(3)]],
                         constant float&     eps [[buffer(4)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[TG / 32];
    device const float* xr = x + row * n;
    float ss = 0.0f;
    for (uint i = tid; i < n; i += TG) {
        ss += xr[i] * xr[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        ss = lane < TG / 32 ? partial[lane] : 0.0f;
        ss = simd_sum(ss);
        if (lane == 0) partial[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = 1.0f / precise::sqrt(partial[0] / float(n) + eps);
    device float* orow = out + row * n;
    for (uint i = tid; i < n; i += TG) {
        orow[i] = xr[i] * scale * w[i];
    }
}
