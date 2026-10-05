// GEMM tiled para prefill: y[t, r] = Σ_k W[r, k] · x[t, k], con W cuantizado (q4_0 / q8_0).
// Cada threadgroup (4 simdgroups, 128 hilos) calcula un bloque de BM = 64 filas × BN = 32 tokens.
// Por cada bloque de 32 a lo largo de K: decuantiza W[64×32] y copia x[32×32] a memoria
// threadgroup, y cada simdgroup acumula 16 filas × 32 tokens con simdgroup_float8x8 (f32).
// Requiere filas % 64 == 0 y columnas % 32 == 0. Tokens arbitrarios (el borde se rellena con 0).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant uint BM = 64;
constant uint BN = 32;
constant uint BK = 32;

// Decuantiza la mitad `part` (0 o 1) de un bloque a 16 floats en `dst` (fila de As, BK floats).
inline void dequant_half(device const block_q4_0& b, uint part, threadgroup float* dst) {
    float d = float(b.d);
    for (uint j = 0; j < 8; ++j) {
        uchar q = b.qs[part * 8 + j];
        dst[part * 8 + j]      = d * float(int(q & 0x0F) - 8);
        dst[part * 8 + j + 16] = d * float(int(q >> 4) - 8);
    }
}

inline void dequant_half(device const block_q8_0& b, uint part, threadgroup float* dst) {
    float d = float(b.d);
    for (uint j = 0; j < 16; ++j) {
        dst[part * 16 + j] = d * float(b.qs[part * 16 + j]);
    }
}

template <typename Block>
void gemm_tiled(device const Block* w, device const float* x, device float* y,
                uint rows, uint cols, uint tokens,
                threadgroup float* As, threadgroup float* Bs,
                uint2 tg, uint tid, uint sg) {
    uint r0 = tg.x * BM;
    uint t0 = tg.y * BN;
    uint nb = cols / BK;

    simdgroup_float8x8 c[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint n = 0; n < 4; ++n)
            c[i][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Reparto de la carga: A (64 filas × 2 mitades = 128 hilos), B (32 tokens × 4 partes).
    uint a_row = tid / 2, a_part = tid % 2;
    uint b_tok = tid / 4, b_part = tid % 4;
    bool b_valid = t0 + b_tok < tokens;
    device const float* xrow = x + (t0 + b_tok) * cols + b_part * 8;

    for (uint kb = 0; kb < nb; ++kb) {
        dequant_half(w[(r0 + a_row) * nb + kb], a_part, As + a_row * BK);
        threadgroup float* bdst = Bs + b_tok * BK + b_part * 8;
        if (b_valid) {
            device const float4* src = (device const float4*)(xrow + kb * BK);
            float4 v0 = src[0], v1 = src[1];
            bdst[0] = v0.x; bdst[1] = v0.y; bdst[2] = v0.z; bdst[3] = v0.w;
            bdst[4] = v1.x; bdst[5] = v1.y; bdst[6] = v1.z; bdst[7] = v1.w;
        } else {
            for (uint i = 0; i < 8; ++i) bdst[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k8 = 0; k8 < BK / 8; ++k8) {
            simdgroup_float8x8 a0, a1;
            simdgroup_load(a0, As + (sg * 16 + 0) * BK + k8 * 8, BK);
            simdgroup_load(a1, As + (sg * 16 + 8) * BK + k8 * 8, BK);
            for (uint n = 0; n < 4; ++n) {
                simdgroup_float8x8 b;
                // Bs es [token][k]; se carga transpuesta para tener [k][token].
                simdgroup_load(b, Bs + (n * 8) * BK + k8 * 8, BK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(c[0][n], a0, b, c[0][n]);
                simdgroup_multiply_accumulate(c[1][n], a1, b, c[1][n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Resultado a memoria threadgroup (reusa As: 64 filas × 32 tokens) y de ahí a y,
    // escribiendo solo los tokens válidos, con accesos contiguos a lo largo de las filas.
    threadgroup float* Cs = As;
    for (uint i = 0; i < 2; ++i)
        for (uint n = 0; n < 4; ++n)
            simdgroup_store(c[i][n], Cs + (sg * 16 + i * 8) * BN + n * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e % BM, t = e / BM;
        if (t0 + t < tokens) y[(t0 + t) * rows + r0 + r] = Cs[r * BN + t];
    }
}

kernel void gemm_tiled_q4_0_f32(device const block_q4_0* w [[buffer(0)]],
                                device const float*      x [[buffer(1)]],
                                device float*            y [[buffer(2)]],
                                constant uint& rows   [[buffer(3)]],
                                constant uint& cols   [[buffer(4)]],
                                constant uint& tokens [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BN * BK];
    gemm_tiled(w, x, y, rows, cols, tokens, As, Bs, tg, tid, sg);
}

kernel void gemm_tiled_q8_0_f32(device const block_q8_0* w [[buffer(0)]],
                                device const float*      x [[buffer(1)]],
                                device float*            y [[buffer(2)]],
                                constant uint& rows   [[buffer(3)]],
                                constant uint& cols   [[buffer(4)]],
                                constant uint& tokens [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BN * BK];
    gemm_tiled(w, x, y, rows, cols, tokens, As, Bs, tg, tid, sg);
}
