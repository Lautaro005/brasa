// Acceso a la KV cache según su tipo (ADR 0009). El host antepone este archivo a los kernels de
// atención, con `#define KV_F16 1` o `#define KV_Q8 1` delante (f32 si no hay ninguno).
// La caché es [cap, hkv] filas de KV_D = 128 valores; fila = posición · hkv + cabeza KV.
//   kv_row(p, fila)  puntero a la fila;
//   kv4(fila, i)     valores 4i .. 4i + 3 como float4;
//   kv1(fila, d)     valor d;
//   load_kv(...)     fragmento 8×8 (claves × dimensiones, o transpuesto) como simdgroup_float8x8.
// Q8: cada fila son 128 int8 y 4 escalas f16 (136 bytes); el valor es escala · q, exacto en f32.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint KV_D = 128;

#if defined(KV_Q8)
typedef uchar KV_T;
constant uint KV_ROW = 136;
inline device const KV_T* kv_row(device const KV_T* p, uint row) { return p + row * KV_ROW; }
inline float kv_scale(device const KV_T* r, uint block) {
    return float(((device const half*)(r + KV_D))[block]);
}
inline float4 kv4(device const KV_T* r, uint i) {
    return float4(((device const char4*)r)[i]) * kv_scale(r, i / 8);
}
inline float kv1(device const KV_T* r, uint d) {
    return float(((device const char*)r)[d]) * kv_scale(r, d / 32);
}
#else
#if defined(KV_F16)
typedef half  KV_T;
typedef half4 KV_T4;
#else
typedef float  KV_T;
typedef float4 KV_T4;
#endif
inline device const KV_T* kv_row(device const KV_T* p, uint row) { return p + row * KV_D; }
inline float4 kv4(device const KV_T* r, uint i) { return float4(((device const KV_T4*)r)[i]); }
inline float kv1(device const KV_T* r, uint d) { return float(r[d]); }
#endif

// Fragmento 8×8 de la cabeza kh: claves j .. j + 8, dimensiones d0 .. d0 + 8. Sin transponer es
// [clave][dimensión] (V); transpuesto, [dimensión][clave] (Kᵀ).
inline simdgroup_float8x8 load_kv(device const KV_T* cache, uint j, uint hkv, uint kh, uint d0,
                                  bool transpose, ushort lane) {
    simdgroup_float8x8 f;
#if defined(KV_Q8)
    // Cada lane arma sus dos elementos: fila fm, columnas fn y fn + 1 (layout de Apple).
    ushort qid = lane / 4;
    ushort fm = (qid & 4) + ((lane / 2) % 4);
    ushort fn = (qid & 2) * 2 + (lane % 2) * 2;
    if (transpose) {
        f.thread_elements()[0] = kv1(kv_row(cache, (j + fn) * hkv + kh), d0 + fm);
        f.thread_elements()[1] = kv1(kv_row(cache, (j + fn + 1) * hkv + kh), d0 + fm);
    } else {
        device const KV_T* r = kv_row(cache, (j + fm) * hkv + kh);
        f.thread_elements()[0] = kv1(r, d0 + fn);
        f.thread_elements()[1] = kv1(r, d0 + fn + 1);
    }
#else
    device const KV_T* p = kv_row(cache, j * hkv + kh) + d0;
    ulong stride = hkv * KV_D;
#if defined(KV_F16)
    simdgroup_half8x8 h;
    simdgroup_load(h, p, stride, ulong2(0, 0), transpose);
    f.thread_elements()[0] = float(h.thread_elements()[0]);
    f.thread_elements()[1] = float(h.thread_elements()[1]);
#else
    simdgroup_load(f, p, stride, ulong2(0, 0), transpose);
#endif
#endif
    return f;
}
