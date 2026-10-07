// GEMM tiled para prefill: y[t, r] = Σ_k W[r, k] · x[t, k], con W cuantizado (q4_0 / q8_0).
// Cada threadgroup (4 simdgroups, 128 hilos) calcula un bloque de 64 filas × 32 tokens; cada
// simdgroup, 32 filas × 16 tokens (8 acumuladores simdgroup_float8x8). Por cada bloque de 32 a lo
// largo de K, cada hilo decuantiza 16 pesos de una fila y copia 8 valores de x a memoria
// threadgroup en el tipo TA, y cada simdgroup hace 8 productos 8×8×8 por paso de 8 en K
// (ADR 0011 y 0030):
// - se calcula Cᵀ = X · Wᵀ (tokens × filas), así cada fragmento se escribe directo en y;
// - los fragmentos de un paso se cargan todos antes de los productos;
// - sin prefetch a registros: con menos registros por hilo entran más threadgroups por núcleo
//   (medido en M1 Pro: el prefetch rinde ~15 % menos);
// - q4_0 se decuantiza con máscaras sobre pares de bytes (sin desplazamientos).
// TA = half (ruta caliente: pesos y activaciones redondeados a f16, acumulación f32, ADR 0030) o
// float (exacto: mismo resultado que el producto en f32 en este orden de suma).
// Requiere filas % 64 == 0 y columnas % 32 == 0. Tokens arbitrarios.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q4_0 { half d; uchar qs[16]; };
struct block_q8_0 { half d; char qs[32]; };

constant ushort BM = 64;
constant ushort BN = 32;

// Posición en As del peso (fila r, k) del bloque: fragmentos 8×8 [k/8][r/8] guardados como [k][r].
inline ushort a_idx(ushort r, ushort k) { return 64 * ((k / 8) * 8 + r / 8) + 8 * (k % 8) + r % 8; }

// Decuantiza los pesos k = 16·il .. 16·il + 15 del bloque b (fila r) a As.
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

template <typename TA, typename Block>
void gemm_tiled(device const Block* w, device const float* x, device float* y,
                uint rows, uint cols, uint tokens,
                threadgroup TA* As, threadgroup TA* Bs,
                uint2 tg, ushort tid, ushort sg) {
    typedef TA TA8 __attribute__((ext_vector_type(8)));
    const uint r0 = tg.x * BM;
    const uint t0 = tg.y * BN;
    const uint nb = cols / 32;

    // W: el hilo decuantiza la mitad il de la fila ar. x: 8 valores (k = 8·bp ..) del token bt.
    const ushort ar = tid / 2, il = tid % 2;
    const ushort bt = tid / 4, bp = tid % 4;
    device const Block* wp = w + (r0 + ar) * nb;
    device const float* xp = x + min(t0 + bt, tokens - 1) * cols + 8 * bp;
    // Bs: fragmentos [k/8][t/8] guardados como [t][k].
    threadgroup TA* bdst = Bs + 64 * (4 * bp + bt / 8) + 8 * (bt % 8);

    simdgroup_matrix<TA, 8, 8> ma[4];
    simdgroup_matrix<TA, 8, 8> mb[2];
    simdgroup_float8x8 mc[8];
    for (ushort i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint kb = 0; kb < nb; ++kb) {
        const float4 x0 = *(device const float4*)(xp + kb * 32);
        const float4 x1 = *(device const float4*)(xp + kb * 32 + 4);
        dequant16(wp + kb, il, As, ar);   // incluye la barrera antes de escribir As
        *(threadgroup TA8*)bdst = TA8(TA(x0[0]), TA(x0[1]), TA(x0[2]), TA(x0[3]),
                                      TA(x1[0]), TA(x1[1]), TA(x1[2]), TA(x1[3]));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const TA* la = As + 4 * 64 * (sg % 2);
        threadgroup const TA* lb = Bs + 2 * 64 * (sg / 2);
#pragma unroll
        for (ushort ik = 0; ik < 4; ik++) {
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
    // 4 KiB, sobre As y Bs que son contiguos) y solo los tokens válidos.
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
    /* As y Bs contiguos: la salida del bloque incompleto usa 4 KiB desde As. */              \
    threadgroup float tgm[(BM * 32 + BN * 32) * sizeof(TA) / 4 < 1024 ? 1024                 \
                          : (BM * 32 + BN * 32) * sizeof(TA) / 4];                           \
    threadgroup TA* As = (threadgroup TA*)tgm;                                                \
    gemm_tiled<TA>(w, x, y, rows, cols, tokens, As, As + BM * 32, tg, tid, sg);               \
}

GEMM_KERNEL(gemm_tiled_q4_0_f16, half, block_q4_0)
GEMM_KERNEL(gemm_tiled_q8_0_f16, half, block_q8_0)
GEMM_KERNEL(gemm_tiled_q4_0_f32, float, block_q4_0)
GEMM_KERNEL(gemm_tiled_q8_0_f32, float, block_q8_0)
