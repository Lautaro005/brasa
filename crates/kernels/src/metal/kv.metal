// Escritura en la KV cache (ADR 0009): convierte el scratch f32 [n] al tipo de la caché.
// En f16 redondea al par más cercano (conversión de Metal), igual que la referencia.
#include <metal_stdlib>
using namespace metal;

kernel void store_kv_f32(device const float* src [[buffer(0)]],
                         device float*       dst [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}

kernel void store_kv_f16(device const float* src [[buffer(0)]],
                         device half*        dst [[buffer(1)]],
                         constant uint& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = half(src[i]);
}
