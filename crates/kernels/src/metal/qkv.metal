// QK-norm + RoPE + escritura de K y V en la caché, en un solo dispatch (T3.5). Reemplaza la
// secuencia rms_norm(q), rms_norm(k), rope(q), rope(k), store_kv(k), store_kv(v) y da los mismos
// bits: la reducción de Σx² sigue el orden de rms_norm_f32 (sumas por simdgroup y suma de las
// parciales en el simdgroup 0), RoPE usa las mismas expresiones y la escritura en la caché, las
// de store_kv. head_dim = 128 (un hilo por elemento).
// Threadgroup g: token t = g / (hq + 2·hkv); dentro del token, primero las hq cabezas de Q,
// después las hkv de K y al final las hkv de V.
// KV_T, KV_D y el tipo de caché vienen de kv_access.metal (ADR 0009).

// Norma RMS de la fila de 128 valores que tiene este threadgroup (un valor por hilo).
inline float rms_scale(float v, threadgroup float* partial, uint tid, uint lane, uint sg, float eps) {
    float ss = simd_sum(v * v);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        ss = lane < 4 ? partial[lane] : 0.0f;
        ss = simd_sum(ss);
        if (lane == 0) partial[4] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return 1.0f / precise::sqrt(partial[4] / float(KV_D) + eps);
}

// Escribe el valor v (elemento tid de la fila) en la fila `row` de la caché.
inline void store_row(device uchar* cache, uint row, float v, uint tid, uint lane, uint sg) {
#if defined(KV_Q8)
    // Un bloque de 32 = un simdgroup (mismas operaciones que store_kv_q8).
    float amax = simd_max(fabs(v));
    half dh = half(precise::divide(amax, 127.0f));
    float d = float(dh);
    float q = (d == 0.0f) ? 0.0f : clamp(rint(precise::divide(v, d)), -127.0f, 127.0f);
    device uchar* r = cache + row * 136;
    ((device char*)r)[tid] = char(q);
    if (lane == 0) ((device half*)(r + 128))[sg] = dh;
#elif defined(KV_F16)
    ((device half*)cache)[row * KV_D + tid] = half(v);
#else
    ((device float*)cache)[row * KV_D + tid] = v;
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
                               uint g    [[threadgroup_position_in_grid]],
                               uint tid  [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[5];
    threadgroup float row[KV_D];
    uint per_tok = hq + 2 * hkv;
    uint t = g / per_tok, r = g % per_tok;

    if (r >= hq + hkv) {
        // V: solo la escritura en la caché.
        uint vh = t * hkv + (r - hq - hkv);
        store_row(vc, vh, v[vh * KV_D + tid], tid, lane, sg);
        return;
    }
    bool is_q = r < hq;
    uint idx = is_q ? t * hq + r : t * hkv + (r - hq);
    device float* x = is_q ? q + idx * KV_D : k + idx * KV_D;
    device const float* w = is_q ? q_norm : k_norm;

    // QK-norm (como rms_norm_f32): out = x · scale · w.
    float xv = x[tid];
    float scale = rms_scale(xv, partial, tid, lane, sg, eps);
    row[tid] = xv * scale * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RoPE NeoX (como rope_neox_f32): el hilo i < 64 rota el par (i, i + 64).
    constexpr uint H = KV_D / 2;
    uint p = pos0 + t;
    if (tid < H) {
        float c = cos_t[p * H + tid];
        float s = sin_t[p * H + tid];
        float a = row[tid];
        float b = row[tid + H];
        row[tid] = a * c - b * s;
        row[tid + H] = b * c + a * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float out = row[tid];
    x[tid] = out;
    if (!is_q) store_row(kc, idx, out, tid, lane, sg);
}
