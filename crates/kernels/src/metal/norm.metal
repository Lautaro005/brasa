// RMSNorm por filas: out = x / sqrt(mean(x²) + eps) * w. Un threadgroup por fila.
// También sirve para QK-norm (filas = tokens × cabezas, n = head_dim).
#include <metal_stdlib>
using namespace metal;

constant uint TG = 256;

template <typename T>
void rms_norm_t(device const float* x,
                         device const float* w,
                         device T*           out,
                         uint n, float eps, uint row, uint tid, uint lane, uint sg,
                         threadgroup float* partial) {
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
    device T* orow = out + row * n;
    for (uint i = tid; i < n; i += TG) {
        orow[i] = T(xr[i] * scale * w[i]);
    }
}


#define RMS_NORM_KERNEL(name, T)                                                              \
kernel void name(device const float* x   [[buffer(0)]],                                      \
                 device const float* w   [[buffer(1)]],                                      \
                 device T*           out [[buffer(2)]],                                      \
                 constant uint&      n   [[buffer(3)]],                                      \
                 constant float&     eps [[buffer(4)]],                                      \
                 uint row  [[threadgroup_position_in_grid]],                                 \
                 uint tid  [[thread_position_in_threadgroup]],                               \
                 uint lane [[thread_index_in_simdgroup]],                                    \
                 uint sg   [[simdgroup_index_in_threadgroup]]) {                             \
    threadgroup float partial[TG / 32];                                                       \
    rms_norm_t(x, w, out, n, eps, row, tid, lane, sg, partial);                               \
}

RMS_NORM_KERNEL(rms_norm_f32, float)
// Salida en f16 para el GEMM de prefill (ADR 0030): mismo valor que rms_norm_f32 redondeado.
RMS_NORM_KERNEL(rms_norm_f16, half)

// Preparación de RMSNorm para decode (T3.5), con la suma residual incluida. Un hilo por elemento,
// threadgroups de 256:
//   x += h (si add != 0);  xw = x · w;  ss[tg] = Σ x² del tramo de 256 elementos del threadgroup.
// El GEMV siguiente (gemv_scaled_*) lee xw, suma los ss en orden fijo y multiplica cada salida por
// 1 / sqrt(Σ ss / n + eps). Así no hace falta el dispatch de RMSNorm, que para una sola fila corre
// en un único threadgroup y está limitado por latencia.
kernel void add_norm_prep(device float*       x   [[buffer(0)]],
                          device const float* h   [[buffer(1)]],
                          device const float* w   [[buffer(2)]],
                          device float*       xw  [[buffer(3)]],
                          device float*       ss  [[buffer(4)]],
                          constant uint&      n   [[buffer(5)]],
                          constant uint&      add [[buffer(6)]],
                          uint i    [[thread_position_in_grid]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    float v = 0.0f;
    if (i < n) {
        v = x[i];
        if (add != 0) {
            v += h[i];
            x[i] = v;
        }
        xw[i] = v * w[i];
    }
    float s = simd_sum(v * v);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        s = lane < 8 ? part[lane] : 0.0f;
        s = simd_sum(s);
        if (lane == 0) ss[tg] = s;
    }
}
