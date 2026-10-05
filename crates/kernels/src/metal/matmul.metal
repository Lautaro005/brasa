// Producto matriz cuantizada × activaciones: y[t, r] = Σ_k W[r, k] · x[t, k].
// W: [rows, cols] en bloques de 32 a lo largo de cols (ADR 0006). Acumulación en f32.
//
// gemv_*: un simdgroup por fila, 4 filas por threadgroup; grid.y recorre tokens (T pequeño).
// gemm_*: un hilo por elemento de salida (versión simple, sin tiling; fase 3 la reemplaza).
#include <metal_stdlib>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };
// q6_0 (ADR 0012): d f16, ql[16] (bits 0-3 de j y j+16), qh[8] (bits 4-5 de j, j+8, j+16, j+24).
struct block_q6_0 { half d; uchar ql[16]; uchar qh[8]; };

constant uint ROWS_PER_TG = 4;

inline float dot_q4_0(device const block_q4_0& b, device const float* x) {
    float s = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        uchar q = b.qs[j];
        s += float(int(q & 0x0F) - 8) * x[j] + float(int(q >> 4) - 8) * x[j + 16];
    }
    return float(b.d) * s;
}

inline float dot_q6_0(device const block_q6_0& b, device const float* x) {
    float s = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        uchar l = b.ql[j], h = b.qh[j % 8];
        uint sh = 2 * (j / 8);
        float lo = float(int((l & 0x0F) | (((h >> sh) & 3) << 4)) - 32);
        float hi = float(int((l >> 4) | (((h >> (sh + 4)) & 3) << 4)) - 32);
        s += lo * x[j] + hi * x[j + 16];
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

// GEMV rápido para decode: cada simdgroup calcula NR = 4 filas y cada threadgroup 2 simdgroups
// (8 filas). Cada lane procesa media fila de bloque (16 pesos): carga una vez los 16 valores de x
// en registros y los usa para las 4 filas; los lanes avanzan de a 16 bloques. Requiere filas % 8.
constant uint NR = 4;

// Escala de RMSNorm a partir de las sumas parciales de add_norm_prep (orden fijo: el mismo en
// todos los threadgroups, así todas las filas usan exactamente la misma escala).
inline float norm_scale(device const float* ss, uint nss, uint n, float eps) {
    float t = 0.0f;
    for (uint i = 0; i < nss; ++i) t += ss[i];
    return 1.0f / precise::sqrt(t / float(n) + eps);
}

// SCALED (gemv_scaled_*, T3.5): x es x · w de add_norm_prep y cada salida se multiplica por la
// escala de RMSNorm.
template <bool SCALED>
void gemv_fast_q4(device const block_q4_0* w, device const float* x, device float* y,
                  uint rows, uint cols, device const float* ss, uint nss, float eps,
                  uint2 tg, uint sg, uint lane) {
    uint row0 = (tg.x * 2 + sg) * NR;
    uint nb = cols / 32;
    uint ix = lane / 2, il = lane % 2;
    device const float* xt = x + tg.y * cols;
    float acc[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib = ix; ib < nb; ib += 16) {
        device const float* xb = xt + ib * 32 + il * 8;
        float xl[8], xh[8];
        for (uint j = 0; j < 8; ++j) { xl[j] = xb[j]; xh[j] = xb[j + 16]; }
        for (uint r = 0; r < NR; ++r) {
            device const block_q4_0& b = w[(row0 + r) * nb + ib];
            device const uchar* qs = b.qs + il * 8;
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uchar q = qs[j];
                s += float(int(q & 0x0F) - 8) * xl[j] + float(int(q >> 4) - 8) * xh[j];
            }
            acc[r] += float(b.d) * s;
        }
    }
    float scale = SCALED ? norm_scale(ss, nss, cols, eps) : 1.0f;
    for (uint r = 0; r < NR; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0) y[tg.y * rows + row0 + r] = SCALED ? v * scale : v;
    }
}

#define GEMV_FAST_ARGS(BLOCK)                                    \
    device const BLOCK* w    [[buffer(0)]],                      \
    device const float* x    [[buffer(1)]],                      \
    device float*       y    [[buffer(2)]],                      \
    constant uint&      rows [[buffer(3)]],                      \
    constant uint&      cols [[buffer(4)]],                      \
    uint2 tg   [[threadgroup_position_in_grid]],                 \
    uint  sg   [[simdgroup_index_in_threadgroup]],               \
    uint  lane [[thread_index_in_simdgroup]]

kernel void gemv_fast_q4_0_f32(GEMV_FAST_ARGS(block_q4_0)) {
    gemv_fast_q4<false>(w, x, y, rows, cols, x, 0, 0.0f, tg, sg, lane);
}

kernel void gemv_scaled_q4_0_f32(GEMV_FAST_ARGS(block_q4_0),
                                 device const float* ss  [[buffer(5)]],
                                 constant uint&      nss [[buffer(6)]],
                                 constant float&     eps [[buffer(7)]]) {
    gemv_fast_q4<true>(w, x, y, rows, cols, ss, nss, eps, tg, sg, lane);
}

kernel void gemv_fast_q8_0_f32(device const block_q8_0* w    [[buffer(0)]],
                               device const float*      x    [[buffer(1)]],
                               device float*            y    [[buffer(2)]],
                               constant uint&           rows [[buffer(3)]],
                               constant uint&           cols [[buffer(4)]],
                               uint2 tg   [[threadgroup_position_in_grid]],
                               uint  sg   [[simdgroup_index_in_threadgroup]],
                               uint  lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg.x * 2 + sg) * NR;
    uint nb = cols / 32;
    uint ix = lane / 2, il = lane % 2;
    device const float* xt = x + tg.y * cols;
    float acc[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib = ix; ib < nb; ib += 16) {
        device const float* xb = xt + ib * 32 + il * 16;
        float xv[16];
        for (uint j = 0; j < 16; ++j) xv[j] = xb[j];
        for (uint r = 0; r < NR; ++r) {
            device const block_q8_0& b = w[(row0 + r) * nb + ib];
            device const char* qs = b.qs + il * 16;
            float s = 0.0f;
            for (uint j = 0; j < 16; ++j) s += float(qs[j]) * xv[j];
            acc[r] += float(b.d) * s;
        }
    }
    for (uint r = 0; r < NR; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0) y[tg.y * rows + row0 + r] = v;
    }
}

// Variantes q6_0 (ADR 0012), para la tabla de embeddings atada (lm_head).
kernel void gemv_q6_0_f32(device const block_q6_0* w    [[buffer(0)]],
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
    device const block_q6_0* wr = w + row * nb;
    device const float* xt = x + tg.y * cols;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32) acc += dot_q6_0(wr[b], xt + b * 32);
    acc = simd_sum(acc);
    if (lane == 0) y[tg.y * rows + row] = acc;
}

kernel void gemm_q6_0_f32(device const block_q6_0* w    [[buffer(0)]],
                          device const float*      x    [[buffer(1)]],
                          device float*            y    [[buffer(2)]],
                          constant uint&           rows [[buffer(3)]],
                          constant uint&           cols [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x, t = gid.y;
    if (row >= rows) return;
    uint nb = cols / 32;
    device const block_q6_0* wr = w + row * nb;
    device const float* xt = x + t * cols;
    float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) acc += dot_q6_0(wr[b], xt + b * 32);
    y[t * rows + row] = acc;
}

// Como gemv_fast_q4_0_f32: cada lane procesa media fila de bloque (elementos 8·il .. 8·il + 8 y
// sus +16) y lee los 8 bytes de qh, donde están los bits altos de los dos grupos.
template <bool SCALED>
void gemv_fast_q6(device const block_q6_0* w, device const float* x, device float* y,
                  uint rows, uint cols, device const float* ss, uint nss, float eps,
                  uint2 tg, uint sg, uint lane) {
    uint row0 = (tg.x * 2 + sg) * NR;
    uint nb = cols / 32;
    uint ix = lane / 2, il = lane % 2;
    device const float* xt = x + tg.y * cols;
    float acc[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib = ix; ib < nb; ib += 16) {
        device const float* xb = xt + ib * 32 + il * 8;
        float xl[8], xh[8];
        for (uint j = 0; j < 8; ++j) { xl[j] = xb[j]; xh[j] = xb[j + 16]; }
        for (uint r = 0; r < NR; ++r) {
            device const block_q6_0& b = w[(row0 + r) * nb + ib];
            device const uchar* ql = b.ql + il * 8;
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uchar l = ql[j], h = b.qh[j];
                int lo = int((l & 0x0F) | (((h >> (2 * il)) & 3) << 4)) - 32;
                int hi = int((l >> 4) | (((h >> (2 * il + 4)) & 3) << 4)) - 32;
                s += float(lo) * xl[j] + float(hi) * xh[j];
            }
            acc[r] += float(b.d) * s;
        }
    }
    float scale = SCALED ? norm_scale(ss, nss, cols, eps) : 1.0f;
    for (uint r = 0; r < NR; ++r) {
        float v = simd_sum(acc[r]);
        if (lane == 0) y[tg.y * rows + row0 + r] = SCALED ? v * scale : v;
    }
}

kernel void gemv_fast_q6_0_f32(GEMV_FAST_ARGS(block_q6_0)) {
    gemv_fast_q6<false>(w, x, y, rows, cols, x, 0, 0.0f, tg, sg, lane);
}

kernel void gemv_scaled_q6_0_f32(GEMV_FAST_ARGS(block_q6_0),
                                 device const float* ss  [[buffer(5)]],
                                 constant uint&      nss [[buffer(6)]],
                                 constant float&     eps [[buffer(7)]]) {
    gemv_fast_q6<true>(w, x, y, rows, cols, ss, nss, eps, tg, sg, lane);
}
