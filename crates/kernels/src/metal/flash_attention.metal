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

// Tipo de la KV cache: el host antepone `#define KV_F16 1` para la variante f16 (ADR 0009).
#ifdef KV_F16
typedef half  KV_T;
typedef half4 KV_T4;
#else
typedef float  KV_T;
typedef float4 KV_T4;
#endif

// Carga un bloque 8×8 de la caché como matriz f32 (en f16 se carga como half y se convierte).
inline simdgroup_float8x8 load_kv(device const KV_T* p, ulong stride, bool transpose) {
    simdgroup_float8x8 f;
#ifdef KV_F16
    simdgroup_half8x8 h;
    simdgroup_load(h, p, stride, ulong2(0, 0), transpose);
    f.thread_elements()[0] = float(h.thread_elements()[0]);
    f.thread_elements()[1] = float(h.thread_elements()[1]);
#else
    simdgroup_load(f, p, stride, ulong2(0, 0), transpose);
#endif
    return f;
}

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
    uint kvstride = hkv * D;

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
    device const KV_T* kbase = k + kh * D;
    device const KV_T* vbase = v + kh * D;

    for (uint j0 = 0; j0 < kend; j0 += BKEYS) {
        // S = Q · Kᵀ (8 × 32)
        simdgroup_float8x8 sf[BKEYS / 8];
        for (uint n = 0; n < BKEYS / 8; ++n) sf[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d8 = 0; d8 < D / 8; ++d8) {
            for (uint n = 0; n < BKEYS / 8; ++n) {
                simdgroup_float8x8 kf = load_kv(kbase + (j0 + n * 8) * kvstride + d8 * 8, kvstride, true);
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
                simdgroup_float8x8 vf = load_kv(vbase + (j0 + n * 8) * kvstride + d8 * 8, kvstride, false);
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

// Variante GQA (la de la ruta caliente). Threadgroup = (bloque de queries, cabeza KV) con
// R = 32 filas: QR = 32 / GQA_G queries de cada una de las GQA_G cabezas de query del grupo
// (fila r: cabeza r / QR, query r % QR). 4 simdgroups. El trabajo se reparte entre simdgroups
// por claves en S = Q·Kᵀ y por dimensiones en O = P·V (como el kernel de llama.cpp), así que
// cada tile de K y de V se lee de memoria del dispositivo una sola vez por threadgroup y cada
// simdgroup guarda en registros solo su cuarto de O. Por cada bloque de BC = 64 claves:
//   1. simdgroup s: S[:, 16s..16s+16) = Q·Kᵀ para las R filas (Q en memoria threadgroup); dos
//      tiles de claves por simdgroup dan 2·R/8 acumuladores independientes;
//   2. softmax online por fila con TPR hilos por fila, máscara causal; P queda en memoria
//      threadgroup y α de cada fila también. Reescalado perezoso (como FlashAttention-3): el
//      máximo de referencia m de una fila solo se actualiza si el máximo nuevo lo supera en más
//      de RESCALE (en log natural); si no, α = 1 y p = exp(s − m) ≤ e^RESCALE. El resultado es el
//      mismo algebraicamente y en f32 no hay riesgo de desborde;
//   3. simdgroup s: O[:, 32s..32s+32) = α·O + P·V; el reescalado de O se saltea cuando todas
//      las filas del simdgroup tienen α = 1 (lo más común después de los primeros bloques).
// Toda la aritmética en f32 (Q no se redondea; K y V f16 se convierten al cargar). La caché debe
// poder leerse hasta el múltiplo de 64 siguiente con valores finitos (KV_ALIGN; las posiciones
// fuera de rango se enmascaran). R = 16 filas deja ~14 KiB de memoria threadgroup: con más filas
// entran menos threadgroups por núcleo y el kernel queda limitado por ocupación (medido en M1 Pro:
// R = 32 rinde ~40 % menos).
#ifndef GQA_G
#define GQA_G 4
#endif
#ifndef FA_ROWS
#define FA_ROWS 16
#endif
constant uint R = FA_ROWS; // filas por threadgroup (16 o 32)
constant uint BC = 64;     // claves por bloque
constant uint NSGF = 4;    // simdgroups
constant float RESCALE = 8.0f;
constant uint KT = BC / 8 / NSGF;   // tiles de claves por simdgroup en S = Q·Kᵀ

[[max_total_threads_per_threadgroup(128)]]
kernel void flash_attn_gqa(device const float* q    [[buffer(0)]],   // [T, hq, D]
                           device const KV_T*  k    [[buffer(1)]],   // [cap, hkv, D]
                           device const KV_T*  v    [[buffer(2)]],   // [cap, hkv, D]
                           device float*       o    [[buffer(3)]],   // [T, hq, D]
                           constant uint& tokens [[buffer(4)]],
                           constant uint& hkv    [[buffer(5)]],
                           constant uint& pos0   [[buffer(6)]],
                           constant float& scale [[buffer(7)]],
                           uint2 tg   [[threadgroup_position_in_grid]],
                           uint  tid  [[thread_index_in_threadgroup]],
                           uint  sg   [[simdgroup_index_in_threadgroup]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint G = GQA_G;
    constexpr uint QR = R / G;                 // queries por threadgroup
    threadgroup float Qs[R * D];               // Q escalada; al final, la salida
    threadgroup float Ss[R * BC];              // S y luego P del bloque
    threadgroup float As[R];                   // α de cada fila en el bloque; al final, 1 / l

    uint kh = tg.y;
    uint hq = hkv * G;
    uint q0 = tg.x * QR;                       // primera query del threadgroup
    uint kvstride = hkv * D;

    for (uint e = tid; e < R * D; e += NSGF * 32) {
        uint r = e / D, d = e % D;
        uint t = q0 + r % QR, h = kh * G + r / QR;
        Qs[e] = (t < tokens) ? q[(t * hq + h) * D + d] * scale : 0.0f;
    }

    // Softmax: TPR hilos consecutivos por fila, KPT claves cada uno.
    constexpr uint TPR = NSGF * 32 / R, KPT = BC / TPR;
    uint sr = tid / TPR, sq = tid % TPR;
    uint spos = pos0 + q0 + sr % QR;           // posición absoluta de la query de la fila sr
    float m = -INFINITY, l = 0.0f;
    // Fila y columnas de los 2 elementos de este lane en cada fragmento 8×8 (layout de Apple).
    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);

    simdgroup_float8x8 of[R / 8][4];           // O[:, 32·sg .. 32·sg + 32)
    for (uint i = 0; i < R / 8; ++i)
        for (uint j = 0; j < 4; ++j) of[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint lk = pos0 + tokens;
    uint kend = pos0 + min(q0 + QR, tokens);   // claves visibles para el threadgroup
    device const KV_T* kbase = k + kh * D;
    device const KV_T* vbase = v + kh * D;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = 0; j0 < kend; j0 += BC) {
        // 1. S[:, 8·KT·sg ..) = Q · Kᵀ
        simdgroup_float8x8 sf[R / 8][KT];
        for (uint i = 0; i < R / 8; ++i)
            for (uint c = 0; c < KT; ++c) sf[i][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d8 = 0; d8 < D / 8; ++d8) {
            simdgroup_float8x8 kf[KT];
            for (uint c = 0; c < KT; ++c)
                kf[c] = load_kv(kbase + (j0 + (sg * KT + c) * 8) * kvstride + d8 * 8, kvstride, true);
            for (uint i = 0; i < R / 8; ++i) {
                simdgroup_float8x8 qf;
                simdgroup_load(qf, Qs + (i * 8) * D + d8 * 8, D);
                for (uint c = 0; c < KT; ++c) simdgroup_multiply_accumulate(sf[i][c], qf, kf[c], sf[i][c]);
            }
        }
        for (uint i = 0; i < R / 8; ++i)
            for (uint c = 0; c < KT; ++c)
                simdgroup_store(sf[i][c], Ss + (i * 8) * BC + (sg * KT + c) * 8, BC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Softmax online de la fila sr sobre las 8 claves de este hilo.
        threadgroup float* srow = Ss + sr * BC + sq * KPT;
        float mx = -INFINITY;
        float sv[KPT];
        for (uint i = 0; i < KPT; ++i) {
            uint j = j0 + sq * KPT + i;
            sv[i] = (j <= spos && j < lk) ? srow[i] : -INFINITY;
            mx = max(mx, sv[i]);
        }
        for (uint x = 1; x < TPR; x <<= 1) mx = max(mx, simd_shuffle_xor(mx, ushort(x)));
        // Reescalado perezoso: m solo cambia si el máximo nuevo lo supera en más de RESCALE.
        float m_new = (mx > m + RESCALE || m == -INFINITY) ? max(m, mx) : m;
        float alpha = (m_new == m) ? 1.0f : precise::exp(m - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < KPT; ++i) {
            float p = (sv[i] == -INFINITY) ? 0.0f : precise::exp(sv[i] - m_new);
            srow[i] = p;
            sum += p;
        }
        for (uint x = 1; x < TPR; x <<= 1) sum += simd_shuffle_xor(sum, ushort(x));
        l = l * alpha + sum;
        m = m_new;
        if (sq == 0) As[sr] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. O[:, 32·sg ..) = α · O + P · V
        for (uint i = 0; i < R / 8; ++i) {
            float a = As[i * 8 + fm];
            if (simd_any(a != 1.0f)) {
                for (uint j = 0; j < 4; ++j) {
                    of[i][j].thread_elements()[0] *= a;
                    of[i][j].thread_elements()[1] *= a;
                }
            }
        }
        for (uint n = 0; n < BC / 8; ++n) {
            simdgroup_float8x8 vf[4];
            for (uint j = 0; j < 4; ++j)
                vf[j] = load_kv(vbase + (j0 + n * 8) * kvstride + sg * 32 + j * 8, kvstride, false);
            for (uint i = 0; i < R / 8; ++i) {
                simdgroup_float8x8 pf;
                simdgroup_load(pf, Ss + (i * 8) * BC + n * 8, BC);
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(of[i][j], pf, vf[j], of[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Salida: O / l por fila, a memoria threadgroup (sobre Qs) y de ahí a o (filas válidas).
    if (sq == 0) As[sr] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < R / 8; ++i) {
        float inv = As[i * 8 + fm];
        for (uint j = 0; j < 4; ++j) {
            of[i][j].thread_elements()[0] *= inv;
            of[i][j].thread_elements()[1] *= inv;
            simdgroup_store(of[i][j], Qs + (i * 8) * D + sg * 32 + j * 8, D);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < R * D; e += NSGF * 32) {
        uint r = e / D, d = e % D;
        uint t = q0 + r % QR, h = kh * G + r / QR;
        if (t < tokens) o[(t * hq + h) * D + d] = Qs[e];
    }
}
