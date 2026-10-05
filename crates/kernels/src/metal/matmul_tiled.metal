// GEMM tiled para prefill: y[t, r] = Σ_k W[r, k] · x[t, k], con W cuantizado (q4_0 / q8_0).
// Cada threadgroup (4 simdgroups, 128 hilos) calcula un bloque de BM = 64 filas × BN = 32 tokens;
// cada simdgroup, 32 filas × 16 tokens (8 acumuladores simdgroup_float8x8, f32). Por cada bloque
// de 32 a lo largo de K: decuantiza W[64×32] y copia x[32×32] a memoria threadgroup, y cada
// simdgroup hace 8 productos 8×8×8 por paso de 8 en K. Medido en M1 Pro (ADR 0011):
// - 16 acumuladores por simdgroup (32×32) hunden el kernel ~10×;
// - la memoria threadgroup va por bloques 8×8 contiguos (sin conflictos de banco);
// - las lecturas de W y x del paso siguiente se emiten antes de calcular el actual.
// Requiere filas % 64 == 0 y columnas % 32 == 0. Tokens arbitrarios (el borde se rellena con 0).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant uint BM = 64;
constant uint BN = 32;
constant uint BK = 32;
constant uint NT = 128;                // hilos por threadgroup
constant uint SGM = 32, SGN = 16;      // sub-bloque por simdgroup (2 × 2 simdgroups)
constant uint FM = SGM / 8, FN = SGN / 8;

// Memoria threadgroup por bloques 8×8 contiguos: el elemento (i, k) de una matriz de BK columnas
// va en ((i / 8) · BK / 8 + k / 8) · 64 + (i % 8) · 8 + k % 8. Un fragmento se carga con stride 8
// y sus 8 filas caen en bancos distintos (con filas de BK = 32 floats caían en los mismos).
inline uint blk(uint i, uint k) { return ((i / 8) * (BK / 8) + k / 8) * 64 + (i % 8) * 8 + k % 8; }

// Decuantiza la mitad `part` (0 o 1) del bloque de la fila r a As.
inline void dequant_half(thread const block_q4_0& b, uint part, threadgroup float* As, uint r) {
    float d = float(b.d);
    for (uint j = 0; j < 8; ++j) {
        uchar q = b.qs[part * 8 + j];
        As[blk(r, part * 8 + j)]      = d * float(int(q & 0x0F) - 8);
        As[blk(r, part * 8 + j + 16)] = d * float(int(q >> 4) - 8);
    }
}

inline void dequant_half(thread const block_q8_0& b, uint part, threadgroup float* As, uint r) {
    float d = float(b.d);
    for (uint j = 0; j < 16; ++j) {
        As[blk(r, part * 16 + j)] = d * float(b.qs[part * 16 + j]);
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
    uint sm = sg % (BM / SGM), sn = sg / (BM / SGM);

    simdgroup_float8x8 c[FM][FN];
    for (uint i = 0; i < FM; ++i)
        for (uint n = 0; n < FN; ++n)
            c[i][n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Cada hilo carga media fila de bloque de W (fila ar) y 8 valores de x (token bt). Las
    // lecturas del paso siguiente se emiten antes de calcular el actual (prefetch a registros).
    uint ar = tid / 2, ap = tid % 2;
    uint bt = tid / 4, bp = tid % 4;
    bool bvalid = t0 + bt < tokens;
    device const Block* wrow = w + (r0 + ar) * nb;
    device const float4* xrow = (device const float4*)(x + (bvalid ? t0 + bt : 0) * cols + bp * 8);
    Block wa = wrow[0];
    float4 x0 = bvalid ? xrow[0] : 0.0f, x1 = bvalid ? xrow[1] : 0.0f;
    // Bs guarda xᵀ por bloques: (k, t) en la posición de blk con BN columnas.
    threadgroup float* bdst = Bs + (bp * (BN / 8) + bt / 8) * 64 + bt % 8;
    threadgroup const float* ab = As + (sm * SGM / 8) * (BK / 8) * 64;
    threadgroup const float* bb = Bs + (sn * SGN / 8) * 64;

    for (uint kb = 0; kb < nb; ++kb) {
        dequant_half(wa, ap, As, ar);
        for (uint j = 0; j < 4; ++j) {
            bdst[j * 8] = x0[j];
            bdst[(j + 4) * 8] = x1[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 1 < nb) {
            wa = wrow[kb + 1];
            if (bvalid) {
                x0 = xrow[(kb + 1) * (BK / 4)];
                x1 = xrow[(kb + 1) * (BK / 4) + 1];
            }
        }

        for (uint k8 = 0; k8 < BK / 8; ++k8) {
            simdgroup_float8x8 b[FN];
            for (uint n = 0; n < FN; ++n) simdgroup_load(b[n], bb + (k8 * (BN / 8) + n) * 64, 8);
            for (uint i = 0; i < FM; ++i) {
                simdgroup_float8x8 a;
                simdgroup_load(a, ab + (i * (BK / 8) + k8) * 64, 8);
                for (uint n = 0; n < FN; ++n) simdgroup_multiply_accumulate(c[i][n], a, b[n], c[i][n]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Salida: C a memoria threadgroup (sobre As: 64 filas × 32 tokens) y de ahí a y, solo
    // tokens válidos, con accesos contiguos a lo largo de las filas.
    threadgroup float* Cs = As;
    for (uint i = 0; i < FM; ++i)
        for (uint n = 0; n < FN; ++n)
            simdgroup_store(c[i][n], Cs + (sm * SGM + i * 8) * BN + sn * SGN + n * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += NT) {
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
