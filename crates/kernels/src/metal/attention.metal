// Atención causal con GQA, versión simple (sin tiling; la fase 3 la reemplaza por una
// estilo FlashAttention). Tres pasos sobre un scratch de puntajes S: [T, Hq, Lk]:
//   1. attn_scores_f32: S[t,h,j] = (q[t,h,:] · k[j, h/G, :]) · scale, -inf si j > pos0 + t
//   2. softmax_f32 por filas de largo Lk (kernel existente)
//   3. attn_pv_f32:     O[t,h,:] = Σ_j S[t,h,j] · v[j, h/G, :]
// q: [T, Hq, D]; k, v: caché [Lk_max, Hkv, D] (posición absoluta j); Lk = pos0 + T.
#include <metal_stdlib>
using namespace metal;

kernel void attn_scores_f32(device const float* q      [[buffer(0)]],
                            device const float* k      [[buffer(1)]],
                            device float*       s      [[buffer(2)]],
                            constant uint&      hq     [[buffer(3)]],
                            constant uint&      hkv    [[buffer(4)]],
                            constant uint&      dim    [[buffer(5)]],
                            constant uint&      pos0   [[buffer(6)]],
                            constant uint&      lk     [[buffer(7)]],
                            constant float&     scale  [[buffer(8)]],
                            uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x, h = gid.y, t = gid.z;
    if (j >= lk || h >= hq) return;
    device float* out = s + (t * hq + h) * lk + j;
    if (j > pos0 + t) {
        *out = -INFINITY;
        return;
    }
    uint kh = h / (hq / hkv);
    device const float* qv = q + (t * hq + h) * dim;
    device const float* kv = k + (j * hkv + kh) * dim;
    float acc = 0.0f;
    for (uint d = 0; d < dim; ++d) acc += qv[d] * kv[d];
    *out = acc * scale;
}

kernel void attn_pv_f32(device const float* p    [[buffer(0)]],
                        device const float* v    [[buffer(1)]],
                        device float*       o    [[buffer(2)]],
                        constant uint&      hq   [[buffer(3)]],
                        constant uint&      hkv  [[buffer(4)]],
                        constant uint&      dim  [[buffer(5)]],
                        constant uint&      pos0 [[buffer(6)]],
                        constant uint&      lk   [[buffer(7)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint d = gid.x, h = gid.y, t = gid.z;
    if (d >= dim || h >= hq) return;
    uint kh = h / (hq / hkv);
    device const float* pr = p + (t * hq + h) * lk;
    uint last = pos0 + t;  // claves válidas: 0..=last (el resto tiene probabilidad 0)
    float acc = 0.0f;
    for (uint j = 0; j <= last; ++j) acc += pr[j] * v[(j * hkv + kh) * dim + d];
    o[(t * hq + h) * dim + d] = acc;
}
