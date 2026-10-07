// GEMM tiled para prefill: y[t, r] = Σ_k W[r, k] · x[t, k], con W cuantizado (q4_0 / q8_0).
// Cada threadgroup (4 simdgroups, 128 hilos) calcula un bloque de 64 filas × 32 tokens; cada
// simdgroup, 32 filas × 16 tokens (8 acumuladores simdgroup_float8x8). Por cada paso de 64 a lo
// largo de K, cada hilo decuantiza un bloque de 32 pesos de una fila y copia 16 valores de x a
// memoria threadgroup en el tipo TA, y cada simdgroup hace 8 productos 8×8×8 por cada 8 de K
// (ADR 0011 y 0030):
// - se calcula Cᵀ = X · Wᵀ (tokens × filas), así cada fragmento se escribe directo en y;
// - los fragmentos de un paso se cargan todos antes de los productos;
// - sin prefetch a registros: con menos registros por hilo entran más threadgroups por núcleo
//   (medido en M1 Pro: el prefetch rinde ~15 % menos);
// - q4_0 se decuantiza con máscaras sobre pares de bytes, sin desplazamientos (+8 %).
// TA = half (ruta caliente: pesos y activaciones redondeados a f16, acumulación f32, ADR 0030) o
// float (exacto: mismo resultado que el producto en f32 en este orden de suma).
// - pasos de 64 en K (dos bloques): la mitad de barreras que con 32 (+2 %).
// Requiere filas % 64 == 0 y columnas % 64 == 0. Tokens arbitrarios.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant ushort BM = 64;
constant ushort BN = 32;
constant ushort BK = 64;

// Posición en As del peso (fila r, k) del bloque: fragmentos 8×8 [k/8][r/8] guardados como [k][r].
inline ushort a_idx(ushort r, ushort k) { return 64 * ((k / 8) * 8 + r / 8) + 8 * (k % 8) + r % 8; }

// Decuantiza el bloque b (32 pesos, k = 32·half .. + 31 del paso) de la fila r a As. La barrera
// va entre la lectura de memoria y la escritura de As (que el paso anterior todavía lee).
template <typename TA>
inline void dequant32(device const block_q4_0* b, ushort half_, threadgroup TA* As, ushort r) {
    // Cada ushort trae los pesos 2j (byte bajo) y 2j + 1 (byte alto) en los nibbles bajos y los
    // pesos 16 + 2j y 17 + 2j en los altos: d·(q & máscara)/2^s − 8d = d·(q − 8), exacto en f32.
    device const ushort* qs = (device const ushort*)b + 1;
    const float d = float(b->d);
    const float md = -8.0f * d;
    ushort q[8];
    for (ushort j = 0; j < 8; ++j) q[j] = qs[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort j = 0; j < 8; ++j) {
        const ushort k = 32 * half_ + 2 * j;
        As[a_idx(r, k)]          = TA(d * float(q[j] & 0x000F) + md);
        As[a_idx(r, k + 1)]      = TA(d / 256.0f * float(q[j] & 0x0F00) + md);
        As[a_idx(r, k + 16)]     = TA(d / 16.0f * float(q[j] & 0x00F0) + md);
        As[a_idx(r, k + 17)]     = TA(d / 4096.0f * float(q[j] & 0xF000) + md);
    }
}

template <typename TA>
inline void dequant32(device const block_q8_0* b, ushort half_, threadgroup TA* As, ushort r) {
    const float d = float(b->d);
    char q[32];   // el bloque mide 34 bytes: sin cargas vectoriales (alineación)
    for (ushort j = 0; j < 32; ++j) q[j] = b->qs[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort j = 0; j < 32; ++j) As[a_idx(r, 32 * half_ + j)] = TA(d * float(q[j]));
}

template <typename TA, typename Block>
void gemm_tiled(device const Block* w, device const float* x, device float* y,
                uint rows, uint cols, uint tokens,
                threadgroup TA* As, threadgroup TA* Bs,
                uint2 tg, ushort tid, ushort sg) {
    typedef TA TA8 __attribute__((ext_vector_type(8)));
    const uint r0 = tg.x * BM;
    const uint t0 = tg.y * BN;
    const uint nb = cols / 32;

    // W: el hilo decuantiza el bloque il (de los dos del paso) de la fila ar. x: 16 valores
    // (k = 16·bp ..) del token bt.
    const ushort ar = tid / 2, il = tid % 2;
    const ushort bt = tid / 4, bp = tid % 4;
    device const Block* wp = w + (r0 + ar) * nb + il;
    device const float* xp = x + min(t0 + bt, tokens - 1) * cols + 16 * bp;
    // Bs: fragmentos [k/8][t/8] guardados como [t][k].
    threadgroup TA* bdst = Bs + 64 * (2 * bp * (BN / 8) + bt / 8) + 8 * (bt % 8);

    simdgroup_matrix<TA, 8, 8> ma[4];
    simdgroup_matrix<TA, 8, 8> mb[2];
    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint kb = 0; kb < nb; kb += 2) {
        float4 xv[4];
        for (ushort j = 0; j < 4; ++j) xv[j] = *(device const float4*)(xp + kb * 32 + 4 * j);
        dequant32(wp + kb, il, As, ar);   // incluye la barrera antes de escribir As
        *(threadgroup TA8*)bdst = TA8(TA(xv[0][0]), TA(xv[0][1]), TA(xv[0][2]), TA(xv[0][3]),
                                      TA(xv[1][0]), TA(xv[1][1]), TA(xv[1][2]), TA(xv[1][3]));
        *(threadgroup TA8*)(bdst + 64 * (BN / 8)) =
            TA8(TA(xv[2][0]), TA(xv[2][1]), TA(xv[2][2]), TA(xv[2][3]),
                TA(xv[3][0]), TA(xv[3][1]), TA(xv[3][2]), TA(xv[3][3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const TA* la = As + 4 * 64 * (sg % 2);
        threadgroup const TA* lb = Bs + 2 * 64 * (sg / 2);
#pragma unroll
        for (ushort ik = 0; ik < BK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (ushort i = 0; i < 4; i++) simdgroup_load(ma[i], la + 64 * i, 8);
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (ushort i = 0; i < 2; i++) simdgroup_load(mb[i], lb + 64 * i, 8);
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (ushort i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            la += 8 * 64;
            lb += 4 * 64;
        }
    }

    // mc[i]: tokens 16·(sg / 2) + 8·(i / 4) .. + 8, filas 32·(sg % 2) + 8·(i % 4) .. + 8.
    if (t0 + BN <= tokens) {
        device float* c = y + (t0 + 16 * (sg / 2)) * rows + r0 + 32 * (sg % 2);
        for (ushort i = 0; i < 8; i++) simdgroup_store(mc[i], c + 8 * (i % 4) + 8 * rows * (i / 4), rows);
        return;
    }
    // Último bloque de tokens incompleto: por memoria threadgroup (16 tokens × 64 filas por pasada,
    // 4 KiB sobre As) y solo los tokens válidos.
    threadgroup float* cs = (threadgroup float*)As;
    for (ushort pass = 0; pass < 2; ++pass) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg / 2 == pass)
            for (ushort i = 0; i < 8; i++)
                simdgroup_store(mc[i], cs + 32 * (sg % 2) + 8 * (i % 4) + 8 * BM * (i / 4), BM);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort e = tid; e < 16 * BM; e += 128) {
            const uint t = t0 + 16 * pass + e / BM;
            if (t < tokens) y[t * rows + r0 + e % BM] = cs[e];
        }
    }
}

#define GEMM_KERNEL(name, TA, Block)                                                          \
kernel void name(device const Block* w [[buffer(0)]],                                        \
                 device const float* x [[buffer(1)]],                                        \
                 device float*       y [[buffer(2)]],                                        \
                 constant uint& rows   [[buffer(3)]],                                        \
                 constant uint& cols   [[buffer(4)]],                                        \
                 constant uint& tokens [[buffer(5)]],                                        \
                 uint2 tg [[threadgroup_position_in_grid]],                                  \
                 ushort tid [[thread_index_in_threadgroup]],                                 \
                 ushort sg  [[simdgroup_index_in_threadgroup]]) {                            \
    threadgroup TA As[BM * BK]; /* ≥ 8 KiB: alcanza para la salida del bloque incompleto */   \
    threadgroup TA Bs[BN * BK];                                                               \
    gemm_tiled<TA>(w, x, y, rows, cols, tokens, As, Bs, tg, tid, sg);                         \
}

GEMM_KERNEL(gemm_tiled_q4_0_f16, half, block_q4_0)
GEMM_KERNEL(gemm_tiled_q8_0_f16, half, block_q8_0)
GEMM_KERNEL(gemm_tiled_q4_0_f32, float, block_q4_0)
GEMM_KERNEL(gemm_tiled_q8_0_f32, float, block_q8_0)
