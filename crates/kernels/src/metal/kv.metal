// Escritura en la KV cache (ADR 0009): convierte el scratch f32 [n] al tipo de la caché.
// En f16 redondea al par más cercano (conversión de Metal), igual que la referencia; Q8, abajo.
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

// Q8 (ADR 0009): un hilo por bloque de 32 valores; n es múltiplo de 128 (filas completas). Cada
// fila de destino son 128 int8 y 4 escalas f16. d = f16(amax / 127), q = rint(x / d) con
// divisiones IEEE, igual que brasa_quant::quantize_kv_q8 y la referencia.
kernel void store_kv_q8(device const float* src [[buffer(0)]],
                        device uchar*       dst [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint b [[thread_position_in_grid]]) {
    if (b * 32 >= n) return;
    device const float* x = src + b * 32;
    float amax = 0.0f;
    for (uint i = 0; i < 32; ++i) amax = max(amax, fabs(x[i]));
    half dh = half(precise::divide(amax, 127.0f));
    float d = float(dh);
    device uchar* row = dst + (b / 4) * 136;
    device char* qs = (device char*)row + (b % 4) * 32;
    for (uint i = 0; i < 32; ++i) {
        float q = (d == 0.0f) ? 0.0f : clamp(rint(precise::divide(x[i], d)), -127.0f, 127.0f);
        qs[i] = char(q);
    }
    ((device half*)(row + 128))[b % 4] = dh;
}
