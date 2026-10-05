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

constant uint D = 128;
constant uint BQ = 32;
constant uint BKEYS = 32;

kernel void flash_attn_f32(device const float* q    [[buffer(0)]],   // [T, hq, D]
                           device const float* k    [[buffer(1)]],   // [cap, hkv, D]
                           device const float* v    [[buffer(2)]],   // [cap, hkv, D]
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
    device const float* kbase = k + kh * D;
    device const float* vbase = v + kh * D;

    for (uint j0 = 0; j0 < kend; j0 += BKEYS) {
        // S = Q · Kᵀ (8 × 32)
        simdgroup_float8x8 sf[BKEYS / 8];
        for (uint n = 0; n < BKEYS / 8; ++n) sf[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d8 = 0; d8 < D / 8; ++d8) {
            for (uint n = 0; n < BKEYS / 8; ++n) {
                simdgroup_float8x8 kf;
                simdgroup_load(kf, kbase + (j0 + n * 8) * kvstride + d8 * 8, kvstride, ulong2(0, 0), true);
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
                simdgroup_float8x8 vf;
                simdgroup_load(vf, vbase + (j0 + n * 8) * kvstride + d8 * 8, kvstride);
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
