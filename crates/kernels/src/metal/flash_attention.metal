// Atención causal con GQA estilo FlashAttention (softmax online), f32, head_dim = 128.
// Threadgroup = (bloque de 32 queries, cabeza h); 4 simdgroups de 8 queries cada uno.
// Por cada bloque de 32 claves: S = Q·Kᵀ (8×32 por simdgroup, simdgroup_float8x8), máscara
// causal, actualización online de (m, l), O = diag(α)·O + P·V. Solo se recorren las claves
// visibles para el bloque (0 ..= pos0 + última query del bloque).
// Requiere que la caché tenga capacidad para leer hasta el múltiplo de 32 siguiente a lk
// (lo garantiza el modelo; las posiciones fuera de rango se enmascaran).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// KV_T, kv_row, kv1 y load_kv vienen de kv_access.metal (lo antepone el host, ADR 0009).

constant uint D = 128;
constant uint BQ = 32;
constant uint BKEYS = 32;

kernel void flash_attn_f32(device const float* q    [[buffer(0)]],   // [T, hq, D]
                           device const KV_T*  k    [[buffer(1)]],   // [cap, hkv, D]
                           device const KV_T*  v    [[buffer(2)]],   // [cap, hkv, D]
                           device float*       o    [[buffer(3)]],   // [T, hq, D]
                           constant uint& tokens [[buffer(4)]],
                           constant uint& hq     [[buffer(5)]],
                           constant uint& hkv    [[buffer(6)]],
                           constant uint& pos0   [[buffer(7)]],
                           constant float& scale [[buffer(8)]],
                           uint2 tg   [[threadgroup_position_in_grid]],
                           uint  tid  [[thread_index_in_threadgroup]],
                           uint  sg   [[simdgroup_index_in_threadgroup]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float Qs[BQ * D];          // queries del bloque (escaladas); luego, salida
    threadgroup float Ss[4 * 8 * BKEYS];   // puntajes / probabilidades por simdgroup
    threadgroup float Ds[4 * 8 * 8];       // diag(α) por simdgroup

    uint t0 = tg.x * BQ;
    uint h = tg.y;
    uint kh = h / (hq / hkv);

    // Q del bloque a memoria threadgroup (filas fuera de rango en 0).
    for (uint e = tid; e < BQ * D; e += 128) {
        uint r = e / D, d = e % D;
        Qs[e] = (t0 + r < tokens) ? q[((t0 + r) * hq + h) * D + d] * scale : 0.0f;
    }
    for (uint e = tid; e < 4 * 64; e += 128) Ds[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 qf[D / 8];
    for (uint d8 = 0; d8 < D / 8; ++d8) simdgroup_load(qf[d8], Qs + (sg * 8) * D + d8 * 8, D);
    simdgroup_float8x8 of[D / 8];
    for (uint d8 = 0; d8 < D / 8; ++d8) of[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float* S = Ss + sg * 8 * BKEYS;
    threadgroup float* Dg = Ds + sg * 64;
    // Cada fila de 8 la atienden 4 lanes, con 8 claves cada uno.
    uint row = lane / 4, quarter = lane % 4;
    uint qpos = pos0 + t0 + sg * 8 + row;          // posición absoluta de la query
    float m = -INFINITY, l = 0.0f;

    uint last_t = min(t0 + BQ, tokens) - 1;
    uint kend = pos0 + last_t + 1;                 // claves visibles para el bloque: [0, kend)

    for (uint j0 = 0; j0 < kend; j0 += BKEYS) {
        // S = Q · Kᵀ (8 × 32)
        simdgroup_float8x8 sf[BKEYS / 8];
        for (uint n = 0; n < BKEYS / 8; ++n) sf[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d8 = 0; d8 < D / 8; ++d8) {
            for (uint n = 0; n < BKEYS / 8; ++n) {
                simdgroup_float8x8 kf = load_kv(k, j0 + n * 8, hkv, kh, d8 * 8, true, lane);
                simdgroup_multiply_accumulate(sf[n], qf[d8], kf, sf[n]);
            }
        }
        for (uint n = 0; n < BKEYS / 8; ++n) simdgroup_store(sf[n], S + n * 8, BKEYS);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax online sobre las 8 claves de este lane.
        threadgroup float* srow = S + row * BKEYS + quarter * 8;
        float mx = -INFINITY;
        for (uint i = 0; i < 8; ++i) {
            uint j = j0 + quarter * 8 + i;
            float s = (j <= qpos && j < kend) ? srow[i] : -INFINITY;
            srow[i] = s;
            mx = max(mx, s);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        float m_new = max(m, mx);
        float alpha = (m_new == -INFINITY) ? 1.0f : precise::exp(m - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            float p = (srow[i] == -INFINITY) ? 0.0f : precise::exp(srow[i] - m_new);
            srow[i] = p;
            sum += p;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = m_new;
        if (quarter == 0) Dg[row * 8 + row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(α) · O + P · V
        simdgroup_float8x8 dg;
        simdgroup_load(dg, Dg, 8);
        simdgroup_float8x8 pf[BKEYS / 8];
        for (uint n = 0; n < BKEYS / 8; ++n) simdgroup_load(pf[n], S + n * 8, BKEYS);
        for (uint d8 = 0; d8 < D / 8; ++d8) {
            simdgroup_float8x8 acc;
            simdgroup_multiply(acc, dg, of[d8]);
            for (uint n = 0; n < BKEYS / 8; ++n) {
                simdgroup_float8x8 vf = load_kv(v, j0 + n * 8, hkv, kh, d8 * 8, false, lane);
                simdgroup_multiply_accumulate(acc, pf[n], vf, acc);
            }
            of[d8] = acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Salida: O / l, solo filas válidas. Se reutiliza la zona de Qs de este simdgroup.
    threadgroup float* Os = Qs + (sg * 8) * D;
    for (uint d8 = 0; d8 < D / 8; ++d8) simdgroup_store(of[d8], Os + d8 * 8, D);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // Cada lane conoce l de su fila; los 4 lanes de la fila escriben 32 columnas cada uno.
    uint t = t0 + sg * 8 + row;
    if (t < tokens) {
        float inv = 1.0f / l;
        for (uint i = 0; i < D / 4; ++i) {
            uint d = quarter * (D / 4) + i;
            o[(t * hq + h) * D + d] = Os[row * D + d] * inv;
        }
    }
}

// Variante GQA (la de la ruta caliente, ADR 0030). Threadgroup = 8 queries de FH cabezas del
// mismo grupo GQA (FH = 2, o 1 sin GQA): 4 simdgroups, bloques de 64 claves (el reparto de
// trabajo de `kernel_flash_attn_ext` de llama.cpp, con K y V compartidos entre cabezas):
//   1. S = Q·Kᵀ: cada simdgroup, los tiles de claves s y s + 4 para las FH cabezas (cada
//      fragmento de K se usa FH veces). Q (escalada) está en memoria threadgroup en half (float
//      con FA_Q_F32, la variante exacta); K se
//      lee directo de la caché como fragmento (half con KV f16, float con f32 y Q8); el producto
//      acumula en f32;
//   2. softmax online en f32: cada simdgroup atiende FH·2 filas y cada lane 2 claves; P queda en
//      memoria threadgroup (f32) y O (en memoria threadgroup, f32) se reescala por fila;
//   3. O[:, 32·s ..) += P · V para las FH cabezas, cada fragmento de V leído una vez.
// Redondeos: los de la caché, Q a f16 y la salida a f16 (ADR 0030: ~5e-4 · max|o| contra Q y
// salida en f32; la salida solo la lee el GEMM de o_proj, que igual la redondea a f16). Con
// FA_Q_F32 (variante exacta), Q y la salida en f32. Medido en
// M1 Pro, T = 512, KV f16 a 16K: 2,48 TFLOPS (Q en f32: 2,21; una cabeza por threadgroup y Q en
// f32: 2,08; el kernel de ADR 0010: ~1,2). Guardar O en registros rinde ~25 % menos que en
// memoria threadgroup; 4 cabezas por threadgroup (32 KiB de memoria threadgroup), ~20 % menos.
// La caché debe poder leerse hasta el múltiplo de 64 siguiente con valores finitos (KV_ALIGN;
// las posiciones fuera de rango se enmascaran).
#ifndef GQA_G
#define GQA_G 4
#endif
constant ushort FH = GQA_G >= 2 ? 2 : 1;   // cabezas por threadgroup
constant ushort FQ = 8;                   // queries por threadgroup
constant ushort FR = FQ * FH;             // filas
constant ushort BC = 64;                  // claves por bloque
constant ushort NSGF = 4;                 // simdgroups
#if defined(FA_Q_F32)
typedef float q_t;                        // variante exacta (PrefillPrecision::F32)
typedef float4 o4_t;                      // salida en f32
#else
typedef half q_t;
typedef half4 o4_t;                       // salida en f16, para el GEMM de o_proj
#endif

#if defined(KV_F16)
typedef simdgroup_half8x8 kv_frag;
inline kv_frag load_frag(device const KV_T* cache, uint j, uint hkv, uint kh, uint d0, bool tr,
                         ushort lane) {
    kv_frag f;
    simdgroup_load(f, kv_row(cache, j * hkv + kh) + d0, ulong(hkv * KV_D), ulong2(0, 0), tr);
    return f;
}
#else
typedef simdgroup_float8x8 kv_frag;
inline kv_frag load_frag(device const KV_T* cache, uint j, uint hkv, uint kh, uint d0, bool tr,
                         ushort lane) {
    return load_kv(cache, j, hkv, kh, d0, tr, lane);
}
#endif

kernel void flash_attn_gqa(device const float* q    [[buffer(0)]],   // [T, hq, D]
                           device const KV_T*  k    [[buffer(1)]],   // [cap, hkv, D]
                           device const KV_T*  v    [[buffer(2)]],   // [cap, hkv, D]
                           device o4_t*        o    [[buffer(3)]],   // [T, hq, D / 4]
                           constant uint& tokens [[buffer(4)]],
                           constant uint& hkv    [[buffer(5)]],
                           constant uint& pos0   [[buffer(6)]],
                           constant float& scale [[buffer(7)]],
                           uint2 tg   [[threadgroup_position_in_grid]],
                           ushort sg   [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup q_t   sq[FR * D];    // Q escalada; fila j: query j % FQ de la cabeza h0 + j / FQ
    threadgroup float ss[FR * BC];   // S y luego P del bloque
    threadgroup float so[FR * D];    // O sin normalizar

    const uint h0 = tg.y * FH;
    const uint hq = hkv * GQA_G;
    const uint kh = h0 / GQA_G;
    const uint q0 = tg.x * FQ;
    constexpr ushort RPS = FR / NSGF;   // filas por simdgroup en el softmax

#pragma unroll
    for (ushort jj = 0; jj < RPS; ++jj) {
        const ushort j = jj * NSGF + sg;
        const uint t = q0 + j % FQ, h = h0 + j / FQ;
        for (ushort i = lane; i < D; i += 32) {
            sq[j * D + i] = t < tokens ? q_t(q[(t * hq + h) * D + i] * scale) : q_t(0);
            so[j * D + i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M[RPS], L[RPS];
#pragma unroll
    for (ushort jj = 0; jj < RPS; ++jj) {
        M[jj] = -FLT_MAX / 2;
        L[jj] = 0.0f;
    }

    const uint kend = pos0 + min(q0 + FQ, tokens);   // claves visibles: [0, kend)
    for (uint j0 = 0; j0 < kend; j0 += BC) {
        // 1. S = Q · Kᵀ
#pragma unroll
        for (ushort cc = 0; cc < BC / 8 / NSGF; ++cc) {
            const ushort tile = cc * NSGF + sg;
            simdgroup_float8x8 mqk[FH];
#pragma unroll
            for (ushort r = 0; r < FH; ++r) mqk[r] = make_filled_simdgroup_matrix<float, 8>(0.0f);
#pragma unroll(4)
            for (ushort i = 0; i < D / 16; ++i) {
                kv_frag mk[2];
                simdgroup_barrier(mem_flags::mem_none);
                mk[0] = load_frag(k, j0 + tile * 8, hkv, kh, 16 * i, true, lane);
                mk[1] = load_frag(k, j0 + tile * 8, hkv, kh, 16 * i + 8, true, lane);
#pragma unroll
                for (ushort r = 0; r < FH; ++r) {
                    simdgroup_matrix<q_t, 8, 8> mq[2];
                    simdgroup_load(mq[0], sq + r * FQ * D + 16 * i, D);
                    simdgroup_load(mq[1], sq + r * FQ * D + 16 * i + 8, D);
                    simdgroup_multiply_accumulate(mqk[r], mq[0], mk[0], mqk[r]);
                    simdgroup_multiply_accumulate(mqk[r], mq[1], mk[1], mqk[r]);
                }
            }
#pragma unroll
            for (ushort r = 0; r < FH; ++r) simdgroup_store(mqk[r], ss + r * FQ * BC + 8 * tile, BC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Softmax online; el lane atiende las claves 2·lane y 2·lane + 1 de cada fila.
#pragma unroll
        for (ushort jj = 0; jj < RPS; ++jj) {
            const ushort j = jj * NSGF + sg;
            const uint qpos = pos0 + q0 + j % FQ;
            threadgroup float2* s2 = (threadgroup float2*)(ss + j * BC) + lane;
            float2 s = *s2;
            const uint key = j0 + 2 * lane;
            if (key > qpos) s[0] = -INFINITY;
            if (key + 1 > qpos) s[1] = -INFINITY;
            const float m = M[jj];
            M[jj] = simd_max(max(m, max(s[0], s[1])));
            const float ms = exp(m - M[jj]);
            const float2 p = exp(s - M[jj]);
            L[jj] = L[jj] * ms + simd_sum(p[0] + p[1]);
            *s2 = p;
            ((threadgroup float4*)(so + j * D))[lane] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. O[:, 32·sg ..) += P · V
        simdgroup_float8x8 lo[FH][4];
#pragma unroll
        for (ushort r = 0; r < FH; ++r)
#pragma unroll
            for (ushort i = 0; i < 4; ++i) simdgroup_load(lo[r][i], so + r * FQ * D + 32 * sg + 8 * i, D);
#pragma unroll(2)
        for (ushort cc = 0; cc < BC / 16; ++cc) {
            simdgroup_float8x8 ps[FH][2];
#pragma unroll
            for (ushort r = 0; r < FH; ++r) {
                simdgroup_load(ps[r][0], ss + r * FQ * BC + 16 * cc, BC);
                simdgroup_load(ps[r][1], ss + r * FQ * BC + 16 * cc + 8, BC);
            }
            const uint jv = j0 + 16 * cc;
#pragma unroll
            for (ushort ii = 0; ii < 2; ++ii) {
                const uint d0 = 32 * sg + 16 * ii;
                kv_frag mv[4];
                mv[0] = load_frag(v, jv, hkv, kh, d0, false, lane);
                mv[1] = load_frag(v, jv, hkv, kh, d0 + 8, false, lane);
                mv[2] = load_frag(v, jv + 8, hkv, kh, d0, false, lane);
                mv[3] = load_frag(v, jv + 8, hkv, kh, d0 + 8, false, lane);
#pragma unroll
                for (ushort r = 0; r < FH; ++r) {
                    simdgroup_multiply_accumulate(lo[r][2 * ii], ps[r][0], mv[0], lo[r][2 * ii]);
                    simdgroup_multiply_accumulate(lo[r][2 * ii + 1], ps[r][0], mv[1], lo[r][2 * ii + 1]);
                    simdgroup_multiply_accumulate(lo[r][2 * ii], ps[r][1], mv[2], lo[r][2 * ii]);
                    simdgroup_multiply_accumulate(lo[r][2 * ii + 1], ps[r][1], mv[3], lo[r][2 * ii + 1]);
                }
            }
        }
#pragma unroll
        for (ushort r = 0; r < FH; ++r)
#pragma unroll
            for (ushort i = 0; i < 4; ++i) simdgroup_store(lo[r][i], so + r * FQ * D + 32 * sg + 8 * i, D);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (ushort jj = 0; jj < RPS; ++jj) {
        const ushort j = jj * NSGF + sg;
        const uint t = q0 + j % FQ, h = h0 + j / FQ;
        if (t < tokens) {
            const float inv = 1.0f / L[jj];
            o[(t * hq + h) * (D / 4) + lane] = o4_t(((threadgroup float4*)(so + j * D))[lane] * inv);
        }
    }
}
