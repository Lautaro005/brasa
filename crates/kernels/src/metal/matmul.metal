// Producto matriz cuantizada × activaciones: y[t, r] = Σ_k W[r, k] · x[t, k].
// W: [rows, cols] en bloques de 32 a lo largo de cols (ADR 0006). Acumulación en f32.
//
// gemv_*: un simdgroup por fila, 4 filas por threadgroup; grid.y recorre tokens (T pequeño).
// gemm_*: un hilo por elemento de salida (versión simple, sin tiling; fase 3 la reemplaza).
#include <metal_stdlib>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant uint ROWS_PER_TG = 4;

inline float dot_q4_0(device const block_q4_0& b, device const float* x) {
    float s = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        uchar q = b.qs[j];
        s += float(int(q & 0x0F) - 8) * x[j] + float(int(q >> 4) - 8) * x[j + 16];
    }
    return float(b.d) * s;
}

inline float dot_q8_0(device const block_q8_0& b, device const float* x) {
    float s = 0.0f;
    for (uint j = 0; j < 32; ++j) {
        s += float(b.qs[j]) * x[j];
    }
    return float(b.d) * s;
}

kernel void gemv_q4_0_f32(device const block_q4_0* w    [[buffer(0)]],
                          device const float*      x    [[buffer(1)]],
                          device float*            y    [[buffer(2)]],
                          constant uint&           rows [[buffer(3)]],
                          constant uint&           cols [[buffer(4)]],
                          uint2 tg   [[threadgroup_position_in_grid]],
                          uint  sg   [[simdgroup_index_in_threadgroup]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    uint row = tg.x * ROWS_PER_TG + sg;
    if (row >= rows) return;
    uint nb = cols / 32;
    device const block_q4_0* wr = w + row * nb;
    device const float* xt = x + tg.y * cols;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) acc += dot_q4_0(wr[b], xt + b * 32);
    acc = simd_sum(acc);
    if (lane == 0) y[tg.y * rows + row] = acc;
}

kernel void gemv_q8_0_f32(device const block_q8_0* w    [[buffer(0)]],
                          device const float*      x    [[buffer(1)]],
                          device float*            y    [[buffer(2)]],
                          constant uint&           rows [[buffer(3)]],
                          constant uint&           cols [[buffer(4)]],
                          uint2 tg   [[threadgroup_position_in_grid]],
                          uint  sg   [[simdgroup_index_in_threadgroup]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    uint row = tg.x * ROWS_PER_TG + sg;
    if (row >= rows) return;
    uint nb = cols / 32;
    device const block_q8_0* wr = w + row * nb;
    device const float* xt = x + tg.y * cols;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) acc += dot_q8_0(wr[b], xt + b * 32);
    acc = simd_sum(acc);
    if (lane == 0) y[tg.y * rows + row] = acc;
}

kernel void gemm_q4_0_f32(device const block_q4_0* w    [[buffer(0)]],
                          device const float*      x    [[buffer(1)]],
                          device float*            y    [[buffer(2)]],
                          constant uint&           rows [[buffer(3)]],
                          constant uint&           cols [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x, t = gid.y;
    if (row >= rows) return;
    uint nb = cols / 32;
    device const block_q4_0* wr = w + row * nb;
    device const float* xt = x + t * cols;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) acc += dot_q4_0(wr[b], xt + b * 32);
    y[t * rows + row] = acc;
}

kernel void gemm_q8_0_f32(device const block_q8_0* w    [[buffer(0)]],
                          device const float*      x    [[buffer(1)]],
                          device float*            y    [[buffer(2)]],
                          constant uint&           rows [[buffer(3)]],
                          constant uint&           cols [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x, t = gid.y;
    if (row >= rows) return;
    uint nb = cols / 32;
    device const block_q8_0* wr = w + row * nb;
    device const float* xt = x + t * cols;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) acc += dot_q8_0(wr[b], xt + b * 32);
    y[t * rows + row] = acc;
}
