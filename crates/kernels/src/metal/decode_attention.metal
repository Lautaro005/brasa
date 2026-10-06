// Atención de decode (1 query) con las claves repartidas en tramos (estilo flash-decoding).
//   attn_decode_partial: grid (tramos, hq). Cada threadgroup (4 simdgroups) recorre un tramo de
//     CHUNK claves; cada simdgroup toma claves alternadas y mantiene un softmax online. Cada lane
//     es dueño de 4 dimensiones (head_dim = 128). Escribe (m, l, acc[128]) por (cabeza, tramo).
//   attn_decode_reduce: grid (hq), 128 hilos: combina los tramos y normaliza.
#include <metal_stdlib>
using namespace metal;

// KV_T, kv_row y kv4 vienen de kv_access.metal (lo antepone el host, ADR 0009).

constant uint D = 128;
constant uint CHUNK = 128;
constant uint PSTRIDE = D + 2;  // m, l, acc[D]

kernel void attn_decode_partial(device const float* q      [[buffer(0)]],  // [hq, D]
                                device const KV_T*  k      [[buffer(1)]],  // [cap, hkv, D]
                                device const KV_T*  v      [[buffer(2)]],
                                device float*       part   [[buffer(3)]],  // [hq, tramos, D+2]
                                constant uint& hq     [[buffer(4)]],
                                constant uint& hkv    [[buffer(5)]],
                                constant uint& lk     [[buffer(6)]],
                                constant float& scale [[buffer(7)]],
                                uint2 tg   [[threadgroup_position_in_grid]],
                                uint  sg   [[simdgroup_index_in_threadgroup]],
                                uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * PSTRIDE];
    uint split = tg.x, h = tg.y;
    uint splits = (lk + CHUNK - 1) / CHUNK;
    uint kh = h / (hq / hkv);
    float4 qv = ((device const float4*)(q + h * D))[lane] * scale;

    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    uint j1 = min(lk, (split + 1) * CHUNK);
    for (uint j = split * CHUNK + sg; j < j1; j += 4) {
        float4 kv = kv4(kv_row(k, j * hkv + kh), lane);
        float s = simd_sum(dot(qv, kv));
        float m_new = max(m, s);
        float alpha = precise::exp(m - m_new);
        float p = precise::exp(s - m_new);
        float4 vv = kv4(kv_row(v, j * hkv + kh), lane);
        acc = acc * alpha + p * vv;
        l = l * alpha + p;
        m = m_new;
    }
    // Combinar los 4 simdgroups del threadgroup.
    threadgroup float* mine = red + sg * PSTRIDE;
    if (lane == 0) { mine[0] = m; mine[1] = l; }
    ((threadgroup float4*)(mine + 2))[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -INFINITY;
        for (uint i = 0; i < 4; ++i) M = max(M, red[i * PSTRIDE]);
        float L = 0.0f;
        float4 A = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float mi = red[i * PSTRIDE];
            float w = (mi == -INFINITY) ? 0.0f : precise::exp(mi - M);
            L += red[i * PSTRIDE + 1] * w;
            A += ((threadgroup float4*)(red + i * PSTRIDE + 2))[lane] * w;
        }
        device float* out = part + (h * splits + split) * PSTRIDE;
        if (lane == 0) { out[0] = M; out[1] = L; }
        ((device float4*)(out + 2))[lane] = A;
    }
}

kernel void attn_decode_reduce(device const float* part [[buffer(0)]],
                               device float*       o    [[buffer(1)]],  // [hq, D]
                               constant uint& lk [[buffer(2)]],
                               uint h [[threadgroup_position_in_grid]],
                               uint d [[thread_position_in_threadgroup]]) {
    uint splits = (lk + CHUNK - 1) / CHUNK;
    device const float* p = part + h * splits * PSTRIDE;
    float M = -INFINITY;
    for (uint s = 0; s < splits; ++s) M = max(M, p[s * PSTRIDE]);
    float L = 0.0f, A = 0.0f;
    for (uint s = 0; s < splits; ++s) {
        float w = precise::exp(p[s * PSTRIDE] - M);
        L += p[s * PSTRIDE + 1] * w;
        A += p[s * PSTRIDE + 2 + d] * w;
    }
    o[h * D + d] = A / L;
}

// Variante "una lane por clave" (la de la ruta caliente): threadgroup = (SG_PER_TG tramos,
// cabeza KV), un simdgroup por tramo. Cada simdgroup atiende a las GQA_G cabezas de query del
// grupo GQA:
//   1. cada lane toma una clave del bloque de 32 y calcula su producto con las GQA_G queries
//      (Q en memoria threadgroup);
//   2. softmax online por bloque: un simd_max y un simd_sum por cabeza cada 32 claves;
//   3. P·V: cada lane es dueño de 4 dimensiones; V se lee coalescido y p se reparte con
//      simd_shuffle.
// K y V se leen una sola vez por grupo, sin staging ni barreras dentro del bucle. El tamaño del
// grupo es constante de compilación (el host antepone `#define GQA_G n`): con un valor en
// tiempo de ejecución los arreglos por cabeza no quedan en registros y el kernel es ~1,5× más
// lento (medido en M1 Pro). Escribe el mismo formato de parciales que attn_decode_partial.
#ifndef GQA_G
#define GQA_G 4
#endif
constant uint SG_PER_TG = 4;

kernel void attn_decode_lanes(device const float* q      [[buffer(0)]],  // [hq, D]
                              device const KV_T*  k      [[buffer(1)]],  // [cap, hkv, D]
                              device const KV_T*  v      [[buffer(2)]],
                              device float*       part   [[buffer(3)]],  // [hq, tramos, D+2]
                              constant uint& hkv    [[buffer(4)]],
                              constant uint& lk     [[buffer(5)]],
                              constant float& scale [[buffer(6)]],
                              uint2 tg   [[threadgroup_position_in_grid]],
                              uint  tid  [[thread_index_in_threadgroup]],
                              uint  sg   [[simdgroup_index_in_threadgroup]],
                              uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint G = GQA_G;
    threadgroup float4 Qs[G * D / 4];
    uint kh = tg.y;
    uint splits = (lk + CHUNK - 1) / CHUNK;
    for (uint e = tid; e < G * D / 4; e += SG_PER_TG * 32) {
        Qs[e] = ((device const float4*)(q + kh * G * D))[e] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint split = tg.x * SG_PER_TG + sg;
    if (split >= splits) return;

    float m[G], l[G];
    float4 acc[G];
    for (uint h = 0; h < G; ++h) { m[h] = -INFINITY; l[h] = 0.0f; acc[h] = 0.0f; }

    uint j1 = min(lk, (split + 1) * CHUNK);
    for (uint j0 = split * CHUNK; j0 < j1; j0 += 32) {
        uint j = j0 + lane;
        bool ok = j < j1;
        float s[G];
        for (uint h = 0; h < G; ++h) s[h] = 0.0f;
        if (ok) {
            device const KV_T* kr = kv_row(k, j * hkv + kh);
#if defined(KV_Q8)
            // Q8: producto con los int8 de cada bloque de 32 y una sola escala por bloque.
            device const char4* kq = (device const char4*)kr;
            // Las lecturas de cada bloque van juntas antes de los productos (ver abajo, P·V).
            for (uint b = 0; b < D / 32; ++b) {
                char4 kc[8];
                for (uint i = 0; i < 8; ++i) kc[i] = kq[b * 8 + i];
                float sc = kv_scale(kr, b);
                float sb[G];
                for (uint h = 0; h < G; ++h) sb[h] = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    float4 kv = float4(kc[i]);
                    for (uint h = 0; h < G; ++h) sb[h] += dot(Qs[h * (D / 4) + b * 8 + i], kv);
                }
                for (uint h = 0; h < G; ++h) s[h] += sb[h] * sc;
            }
#else
            // De a 4 lecturas antes de los productos (ver abajo, P·V).
            for (uint d0 = 0; d0 < D / 4; d0 += 4) {
                float4 kk[4];
                for (uint u = 0; u < 4; ++u) kk[u] = kv4(kr, d0 + u);
                for (uint u = 0; u < 4; ++u)
                    for (uint h = 0; h < G; ++h) s[h] += dot(Qs[h * (D / 4) + d0 + u], kk[u]);
            }
#endif
        }
        float p[G];
        for (uint h = 0; h < G; ++h) {
            float sh = ok ? s[h] : -INFINITY;
            float mn = max(m[h], simd_max(sh));  // finito: la lane 0 siempre tiene clave
            float alpha = precise::exp(m[h] - mn);
            p[h] = ok ? precise::exp(sh - mn) : 0.0f;
            l[h] = l[h] * alpha + simd_sum(p[h]);
            acc[h] *= alpha;
            m[h] = mn;
        }
        // P·V. En bloques completos, las filas de V se leen de a 4 antes de acumular: con una
        // lectura por iteración la latencia de memoria se pagaba en serie (~60 µs fijos por capa
        // en M1 Pro, medido con attn_decode_sweep). Mismo orden de sumas, mismos bits.
        uint n = min(32u, j1 - j0);
        if (n == 32) {
            for (uint jb = 0; jb < 32; jb += 4) {
                float4 vv[4];
                for (uint u = 0; u < 4; ++u) vv[u] = kv4(kv_row(v, (j0 + jb + u) * hkv + kh), lane);
                for (uint u = 0; u < 4; ++u)
                    for (uint h = 0; h < G; ++h) acc[h] += simd_shuffle(p[h], ushort(jb + u)) * vv[u];
            }
        } else for (uint jj = 0; jj < n; ++jj) {
            float4 vv = kv4(kv_row(v, (j0 + jj) * hkv + kh), lane);
            for (uint h = 0; h < G; ++h) acc[h] += simd_shuffle(p[h], ushort(jj)) * vv;
        }
    }
    for (uint h = 0; h < G; ++h) {
        device float* out = part + ((kh * G + h) * splits + split) * PSTRIDE;
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
        ((device float4*)(out + 2))[lane] = acc[h];
    }
}
