// GEMM tiled para prefill: y[t, r] = Σ_k W[r, k] · x[t, k], con W cuantizado (q4_0 / q8_0).
// Cada threadgroup (4 simdgroups, 128 hilos) calcula un bloque de 64 filas × BN tokens (64 en
// los bloques completos, 32 en el resto); cada simdgroup, 32 filas × BN/2 tokens (16 u 8
// acumuladores simdgroup_float8x8). Por cada paso de 32 a lo largo de K, cada hilo decuantiza 16
// pesos de una fila y copia BN/4 valores de x a memoria threadgroup en el tipo TA, y cada
// simdgroup hace BN/4 productos 8×8×8 por cada 8 de K (ADR 0011 y 0030):
// - se calcula Cᵀ = X · Wᵀ (tokens × filas), así cada fragmento se escribe directo en y;
// - los fragmentos de un paso se cargan todos antes de los productos;
// - con 16 acumuladores, cualquier código de borde en la salida (aun con índices constantes) hunde
//   el kernel ~10× (era la causa del derrumbe de ADR 0011): por eso los tokens que no completan
//   un bloque de 64 van a la variante de 32;
// - sin prefetch a registros: con menos registros por hilo entran más threadgroups por núcleo
//   (medido en M1 Pro: el prefetch rinde ~15 % menos);
// - q4_0 se decuantiza con máscaras sobre pares de bytes, sin desplazamientos (+8 %).
// TA = half (ruta caliente: pesos redondeados a f16, acumulación f32, ADR 0030; x llega en f16,
// escrita así por el kernel anterior: RMSNorm, atención o SwiGLU) o float (exacto: x en f32,
// mismo resultado que el producto en f32 en este orden de suma).
// Requiere filas % 64 == 0 y columnas % 32 == 0. Tokens arbitrarios.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant ushort BM = 64;
constant ushort BK = 32;

// Posición en As del peso (fila r, k) del paso: fragmentos 8×8 [k/8][r/8] guardados como [k][r].
inline ushort a_idx(ushort r, ushort k) { return 64 * ((k / 8) * (BM / 8) + r / 8) + 8 * (k % 8) + r % 8; }

// Decuantiza los pesos k = 16·il .. 16·il + 15 del bloque b (fila r) a As. La barrera va entre la
// lectura de memoria y la escritura de As (que el paso anterior todavía lee).
template <typename TA>
inline void dequant16(device const block_q4_0* b, ushort il, threadgroup TA* As, ushort r) {
    // Cada ushort trae los pesos 2j (byte bajo) y 2j + 1 (byte alto); il = 1 toma los nibbles
    // altos (pesos 16..31). d1 · (q & máscara) + md = d · (q − 8), exacto en f32.
    device const ushort* qs = (device const ushort*)b + 1;
    const float d = float(b->d);
    const float d1 = il ? d / 16.0f : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    ushort q[8];
    for (ushort j = 0; j < 8; ++j) q[j] = qs[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort j = 0; j < 8; ++j) {
        const ushort k = 16 * il + 2 * j;
        As[a_idx(r, k)] = TA(d1 * float(q[j] & mask0) + md);
        As[a_idx(r, k + 1)] = TA(d2 * float(q[j] & mask1) + md);
    }
}

template <typename TA>
inline void dequant16(device const block_q8_0* b, ushort il, threadgroup TA* As, ushort r) {
    const float d = float(b->d);
    char q[16];   // el bloque mide 34 bytes: sin cargas vectoriales (alineación)
    for (ushort j = 0; j < 16; ++j) q[j] = b->qs[16 * il + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort j = 0; j < 16; ++j) As[a_idx(r, 16 * il + j)] = TA(d * float(q[j]));
}

// BN = 64: 16 acumuladores por simdgroup (32 filas × 32 tokens), solo bloques de tokens
// completos (sin código de borde: ver el comentario del encabezado). BN = 32: 8 acumuladores
// (32 × 16) y bloque de tokens incompleto; el host lo usa para los tokens que sobran del
// múltiplo de 64.
template <ushort BN, typename TA, typename Block>
void gemm_tiled(device const Block* w, device const TA* x, device float* y,
                uint rows, uint cols, uint tokens,
                threadgroup TA* As, threadgroup TA* Bs,
                uint2 tg, ushort tid, ushort sg) {
    typedef TA TA8 __attribute__((ext_vector_type(8)));
    constexpr ushort FN = BN / 16;           // fragmentos de tokens por simdgroup
    constexpr ushort VPT = BN * BK / 128;    // valores de x por hilo (8 o 16)
    const uint r0 = tg.x * BM;
    const uint t0 = tg.y * BN;
    const uint nb = cols / BK;

    // W: el hilo decuantiza la mitad il del bloque de la fila ar. x: VPT valores
    // (k = VPT·bh ..) del token bt.
    const ushort ar = tid / 2, il = tid % 2;
    const ushort bt = tid / (BK / VPT), bh = tid % (BK / VPT);
    device const Block* wp = w + (r0 + ar) * nb;
    device const TA* xp = x + min(t0 + bt, tokens - 1) * cols + VPT * bh;
    // Bs: fragmentos [k/8][t/8] guardados como [t][k].
    threadgroup TA* bdst = Bs + 64 * ((VPT / 8) * bh * (BN / 8) + bt / 8) + 8 * (bt % 8);

    simdgroup_matrix<TA, 8, 8> ma[4];
    simdgroup_matrix<TA, 8, 8> mb[FN];
    simdgroup_float8x8 mc[4 * FN];
#pragma unroll
    for (ushort i = 0; i < 4 * FN; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint kb = 0; kb < nb; ++kb) {
        typedef TA TA4 __attribute__((ext_vector_type(4)));
        TA4 xv[VPT / 4];
#pragma unroll
        for (ushort j = 0; j < VPT / 4; ++j) xv[j] = *(device const TA4*)(xp + kb * BK + 4 * j);
        dequant16(wp + kb, il, As, ar);   // incluye la barrera antes de escribir As
#pragma unroll
        for (ushort g = 0; g < VPT / 8; ++g)
            *(threadgroup TA8*)(bdst + 64 * (BN / 8) * g) =
                TA8(TA(xv[2 * g][0]), TA(xv[2 * g][1]), TA(xv[2 * g][2]), TA(xv[2 * g][3]),
                    TA(xv[2 * g + 1][0]), TA(xv[2 * g + 1][1]), TA(xv[2 * g + 1][2]),
                    TA(xv[2 * g + 1][3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const TA* la = As + 4 * 64 * (sg % 2);
        threadgroup const TA* lb = Bs + FN * 64 * (sg / 2);
#pragma unroll
        for (ushort ik = 0; ik < BK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (ushort i = 0; i < 4; i++) simdgroup_load(ma[i], la + 64 * i, 8);
#pragma unroll
            for (ushort i = 0; i < FN; i++) simdgroup_load(mb[i], lb + 64 * i, 8);
            simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
            for (ushort i = 0; i < 4 * FN; i++)
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            la += (BM / 8) * 64;
            lb += (BN / 8) * 64;
        }
    }

    // mc[i]: tokens (BN/2)·(sg / 2) + 8·(i / 4) .. + 8, filas 32·(sg % 2) + 8·(i % 4) .. + 8.
    if (BN == 64 || t0 + BN <= tokens) {
        device float* c = y + (t0 + (BN / 2) * (sg / 2)) * rows + r0 + 32 * (sg % 2);
#pragma unroll
        for (ushort i = 0; i < 4 * FN; i++)
            simdgroup_store(mc[i], c + 8 * (i % 4) + 8 * rows * (i / 4), rows);
        return;
    }
    // Bloque de tokens incompleto (solo BN = 32): por memoria threadgroup (16 tokens × 64 filas
    // por pasada, 4 KiB desde As) y solo los tokens válidos.
    threadgroup float* cs = (threadgroup float*)As;
    for (ushort pass = 0; pass < 2; ++pass) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg / 2 == pass) {
#pragma unroll
            for (ushort i = 0; i < 4 * FN; i++)
                simdgroup_store(mc[i], cs + 32 * (sg % 2) + 8 * (i % 4) + 8 * BM * (i / 4), BM);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort e = tid; e < 16 * BM; e += 128) {
            const uint t = t0 + 16 * pass + e / BM;
            if (t < tokens) y[t * rows + r0 + e % BM] = cs[e];
        }
    }
}

// Hasta tres matrices con la misma entrada x (q, k, v o gate, up) en un dispatch: el eje x de la
// grilla recorre los bloques de 64 filas de w0, w1 y w2 seguidos (r1 = r2 = 0 para una sola).
#define GEMM_KERNEL(name, BN, TA, Block)                                                      \
kernel void name(device const Block* w0 [[buffer(0)]],                                       \
                 device const Block* w1 [[buffer(1)]],                                       \
                 device const Block* w2 [[buffer(2)]],                                       \
                 device const TA*    x  [[buffer(3)]],                                       \
                 device float*       y0 [[buffer(4)]],                                       \
                 device float*       y1 [[buffer(5)]],                                       \
                 device float*       y2 [[buffer(6)]],                                       \
                 constant uint& r0      [[buffer(7)]],                                       \
                 constant uint& r1      [[buffer(8)]],                                       \
                 constant uint& r2      [[buffer(9)]],                                       \
                 constant uint& cols    [[buffer(10)]],                                      \
                 constant uint& tokens  [[buffer(11)]],                                      \
                 uint2 tg [[threadgroup_position_in_grid]],                                  \
                 ushort tid [[thread_index_in_threadgroup]],                                 \
                 ushort sg  [[simdgroup_index_in_threadgroup]]) {                            \
    /* As y Bs contiguos; la salida del bloque incompleto usa 4 KiB desde As. */              \
    threadgroup float tgm[(BM + BN) * BK * sizeof(TA) / 4 < 1024 ? 1024                      \
                          : (BM + BN) * BK * sizeof(TA) / 4];                                \
    threadgroup TA* As = (threadgroup TA*)tgm;                                                \
    const uint b0 = r0 / BM, b1 = r1 / BM;                                                    \
    if (tg.x < b0)                                                                            \
        gemm_tiled<BN>(w0, x, y0, r0, cols, tokens, As, As + BM * BK, tg, tid, sg);          \
    else if (tg.x < b0 + b1)                                                                  \
        gemm_tiled<BN>(w1, x, y1, r1, cols, tokens, As, As + BM * BK,                        \
                       uint2(tg.x - b0, tg.y), tid, sg);                                      \
    else                                                                                      \
        gemm_tiled<BN>(w2, x, y2, r2, cols, tokens, As, As + BM * BK,                        \
                       uint2(tg.x - b0 - b1, tg.y), tid, sg);                                 \
}

GEMM_KERNEL(gemm64_q4_0_f16, 64, half, block_q4_0)
GEMM_KERNEL(gemm64_q8_0_f16, 64, half, block_q8_0)
GEMM_KERNEL(gemm64_q4_0_f32, 64, float, block_q4_0)
GEMM_KERNEL(gemm64_q8_0_f32, 64, float, block_q8_0)
GEMM_KERNEL(gemm32_q4_0_f16, 32, half, block_q4_0)
GEMM_KERNEL(gemm32_q8_0_f16, 32, half, block_q8_0)
GEMM_KERNEL(gemm32_q4_0_f32, 32, float, block_q4_0)
GEMM_KERNEL(gemm32_q8_0_f32, 32, float, block_q8_0)
