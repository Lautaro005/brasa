// QK-norm + RoPE + escritura de K y V en la caché, en un solo dispatch (T3.5). Reemplaza la
// secuencia rms_norm(q), rms_norm(k), rope(q), rope(k), store_kv(k), store_kv(v) y da los mismos
// bits: la reducción de Σx² sigue el orden de rms_norm_f32 (sumas por simdgroup y suma de las
// parciales en el simdgroup 0), RoPE usa las mismas expresiones y la escritura en la caché, las
// de store_kv. head_dim = 128 (un hilo por elemento).
// Threadgroup g: token t = g / (hq + 2·hkv); dentro del token, primero las hq cabezas de Q,
// después las hkv de K y al final las hkv de V.
// KV_T, KV_D y el tipo de caché vienen de kv_access.metal (ADR 0009).

// Un simdgroup por fila, 4 filas por threadgroup (antes: un threadgroup de 128 hilos por fila,
// con barreras; medido en M1 Pro con T = 512: ~2× más lento). El lane l tiene los elementos
// l + 32·j (j = 0..3): las sumas Σx² de cada j son las de los 4 simdgroups de rms_norm_f32 y se
// suman igual (simd_sum de las 4 parciales en los lanes 0..3), y los pares de RoPE (i, i + 64)
// quedan en el mismo lane.

// Escribe el valor v (elemento l + 32·j de la fila `row`) en la caché.
inline void store_elem(device uchar* cache, uint row, float v, uint lane, uint j) {
#if defined(KV_Q8)
    // Un bloque de 32 = los 32 lanes con el mismo j (mismas operaciones que store_kv_q8).
    float amax = simd_max(fabs(v));
    half dh = half(precise::divide(amax, 127.0f));
    float d = float(dh);
    float q = (d == 0.0f) ? 0.0f : clamp(rint(precise::divide(v, d)), -127.0f, 127.0f);
    device uchar* r = cache + row * 136;
    ((device char*)r)[lane + 32 * j] = char(q);
    if (lane == 0) ((device half*)(r + 128))[j] = dh;
#elif defined(KV_F16)
    ((device half*)cache)[row * KV_D + lane + 32 * j] = half(v);
#else
    ((device float*)cache)[row * KV_D + lane + 32 * j] = v;
#endif
}

kernel void qk_norm_rope_store(device float*       q      [[buffer(0)]],   // [T, hq, 128]
                               device float*       k      [[buffer(1)]],   // [T, hkv, 128]
                               device const float* v      [[buffer(2)]],   // [T, hkv, 128]
                               device const float* q_norm [[buffer(3)]],
                               device const float* k_norm [[buffer(4)]],
                               device const float* cos_t  [[buffer(5)]],
                               device const float* sin_t  [[buffer(6)]],
                               device uchar*       kc     [[buffer(7)]],   // caché K desde pos0
                               device uchar*       vc     [[buffer(8)]],   // caché V desde pos0
                               constant uint&      hq     [[buffer(9)]],
                               constant uint&      hkv    [[buffer(10)]],
                               constant uint&      pos0   [[buffer(11)]],
                               constant float&     eps    [[buffer(12)]],
                               constant uint&      rows   [[buffer(13)]],  // T · (hq + 2·hkv)
                               uint g    [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]]) {
    const uint gr = g * 4 + sg;
    if (gr >= rows) return;
    const uint per_tok = hq + 2 * hkv;
    const uint t = gr / per_tok, r = gr % per_tok;

    if (r >= hq + hkv) {
        // V: solo la escritura en la caché.
        const uint vh = t * hkv + (r - hq - hkv);
        for (uint j = 0; j < 4; ++j) store_elem(vc, vh, v[vh * KV_D + lane + 32 * j], lane, j);
        return;
    }
    const bool is_q = r < hq;
    const uint idx = is_q ? t * hq + r : t * hkv + (r - hq);
    device float* x = is_q ? q + idx * KV_D : k + idx * KV_D;
    device const float* w = is_q ? q_norm : k_norm;

    // QK-norm (como rms_norm_f32): out = x · scale · w.
    float xv[4], p[4];
    for (uint j = 0; j < 4; ++j) {
        xv[j] = x[lane + 32 * j];
        p[j] = simd_sum(xv[j] * xv[j]);
    }
    float ss = lane == 0 ? p[0] : lane == 1 ? p[1] : lane == 2 ? p[2] : lane == 3 ? p[3] : 0.0f;
    ss = simd_sum(ss);
    const float scale = 1.0f / precise::sqrt(ss / float(KV_D) + eps);
    float y[4];
    for (uint j = 0; j < 4; ++j) y[j] = xv[j] * scale * w[lane + 32 * j];

    // RoPE NeoX (como rope_neox_f32): pares (l, l + 64) y (l + 32, l + 96).
    constexpr uint H = KV_D / 2;
    const uint pos = pos0 + t;
    for (uint j = 0; j < 2; ++j) {
        const uint i = lane + 32 * j;
        const float c = cos_t[pos * H + i];
        const float s = sin_t[pos * H + i];
        const float a = y[j], b = y[j + 2];
        y[j] = a * c - b * s;
        y[j + 2] = b * c + a * s;
    }
    for (uint j = 0; j < 4; ++j) {
        x[lane + 32 * j] = y[j];
        if (!is_q) store_elem(kc, idx, y[j], lane, j);
    }
}
