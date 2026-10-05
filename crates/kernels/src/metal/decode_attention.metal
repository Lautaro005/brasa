// Atención de decode (1 query) con las claves repartidas en tramos (estilo flash-decoding).
//   attn_decode_partial: grid (tramos, hq). Cada threadgroup (4 simdgroups) recorre un tramo de
//     CHUNK claves; cada simdgroup toma claves alternadas y mantiene un softmax online. Cada lane
//     es dueño de 4 dimensiones (head_dim = 128). Escribe (m, l, acc[128]) por (cabeza, tramo).
//   attn_decode_reduce: grid (hq), 128 hilos: combina los tramos y normaliza.
#include <metal_stdlib>
using namespace metal;

constant uint D = 128;
constant uint CHUNK = 256;
constant uint PSTRIDE = D + 2;  // m, l, acc[D]

kernel void attn_decode_partial(device const float* q      [[buffer(0)]],  // [hq, D]
                                device const float* k      [[buffer(1)]],  // [cap, hkv, D]
                                device const float* v      [[buffer(2)]],
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
    uint kvstride = hkv * D;
    float4 qv = ((device const float4*)(q + h * D))[lane] * scale;

    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    uint j1 = min(lk, (split + 1) * CHUNK);
    for (uint j = split * CHUNK + sg; j < j1; j += 4) {
        float4 kv = ((device const float4*)(k + j * kvstride + kh * D))[lane];
        float s = simd_sum(dot(qv, kv));
        float m_new = max(m, s);
        float alpha = precise::exp(m - m_new);
        float p = precise::exp(s - m_new);
        float4 vv = ((device const float4*)(v + j * kvstride + kh * D))[lane];
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

// Variante GQA: threadgroup = (tramo, cabeza KV) con un simdgroup por cabeza de query del grupo
// (group = hq / hkv ≤ 8). K y V del tramo se copian a memoria threadgroup de a STAGE claves y los
// simdgroups del grupo las leen de ahí: la KV cache se lee una vez por grupo y no una por cabeza.
// Mismo formato de parciales que attn_decode_partial (se reduce con attn_decode_reduce).
constant uint STAGE = 16;

kernel void attn_decode_gqa_partial(device const float* q      [[buffer(0)]],  // [hq, D]
                                    device const float* k      [[buffer(1)]],  // [cap, hkv, D]
                                    device const float* v      [[buffer(2)]],
                                    device float*       part   [[buffer(3)]],  // [hq, tramos, D+2]
                                    constant uint& hq     [[buffer(4)]],
                                    constant uint& hkv    [[buffer(5)]],
                                    constant uint& lk     [[buffer(6)]],
                                    constant float& scale [[buffer(7)]],
                                    uint2 tg   [[threadgroup_position_in_grid]],
                                    uint  tid  [[thread_index_in_threadgroup]],
                                    uint  sg   [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]],
                                    uint2 ntg  [[threads_per_threadgroup]]) {
    threadgroup float4 Ks[STAGE * D / 4];
    threadgroup float4 Vs[STAGE * D / 4];
    uint split = tg.x, kh = tg.y;
    uint group = hq / hkv;
    uint h = kh * group + sg;          // cabeza de query de este simdgroup
    uint splits = (lk + CHUNK - 1) / CHUNK;
    uint kvstride = hkv * D;
    float4 qv = ((device const float4*)(q + h * D))[lane] * scale;

    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    uint j0 = split * CHUNK;
    uint j1 = min(lk, j0 + CHUNK);
    for (uint s0 = j0; s0 < j1; s0 += STAGE) {
        uint n = min(STAGE, j1 - s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < n * D / 4; e += ntg.x) {
            uint key = e / (D / 4), c = e % (D / 4);
            Ks[e] = ((device const float4*)(k + (s0 + key) * kvstride + kh * D))[c];
            Vs[e] = ((device const float4*)(v + (s0 + key) * kvstride + kh * D))[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < n; ++i) {
            float s = simd_sum(dot(qv, Ks[i * (D / 4) + lane]));
            float m_new = max(m, s);
            float alpha = precise::exp(m - m_new);
            float p = precise::exp(s - m_new);
            acc = acc * alpha + p * Vs[i * (D / 4) + lane];
            l = l * alpha + p;
            m = m_new;
        }
    }
    device float* out = part + (h * splits + split) * PSTRIDE;
    if (lane == 0) { out[0] = m; out[1] = l; }
    ((device float4*)(out + 2))[lane] = acc;
}
