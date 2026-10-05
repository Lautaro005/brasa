// Softmax por filas, numéricamente estable: e^(x - max) / Σ e^(x - max). Un threadgroup por fila.
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

kernel void softmax_f32(device const float* x   [[buffer(0)]],
                        device float*       out [[buffer(1)]],
                        constant uint&      n   [[buffer(2)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_position_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[TG / 32];
    device const float* xr = x + row * n;
    device float* orow = out + row * n;

    float m = -INFINITY;
    for (uint i = tid; i < n; i += TG) m = max(m, xr[i]);
    m = simd_max(m);
    if (lane == 0) partial[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        m = lane < TG / 32 ? partial[lane] : -INFINITY;
        m = simd_max(m);
        if (lane == 0) partial[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = 0.0f;
    for (uint i = tid; i < n; i += TG) {
        float e = precise::exp(xr[i] - m);
        orow[i] = e;
        s += e;
    }
    s = simd_sum(s);
    if (lane == 0) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = lane < TG / 32 ? partial[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / partial[0];
    for (uint i = tid; i < n; i += TG) orow[i] *= inv;
}
