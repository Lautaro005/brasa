// KV cache TurboQuant TQ4 (ADR 0033). Especificación: brasa_quant::turbo (quantize_row).
// El host antepone `TQ_CB` y `TQ_MID` (brasa_quant::turbo::metal_constants) a esta fuente.
//
//   tq_rotate_f32 / tq_rotate_f16: x ← R·x (o Rᵀ·x con transpose), por filas de 128 valores,
//     un threadgroup de 128 hilos por fila. Rota Q antes de la atención y la salida después.
//   tq_store_tq4: escribe filas de 128 valores (f32) en la caché de 68 bytes: y = R·x, norma
//     f16 y un índice de 4 bits por coordenada de y / norma.
#include <metal_stdlib>
using namespace metal;

constant uint TQ_D = 128;
constant uint TQ_ROW = 68;

kernel void tq_rotate_f32(device float*       x         [[buffer(0)]],  // [filas, 128]
                          device const float* R         [[buffer(1)]],  // [128, 128]
                          constant uint&      transpose [[buffer(2)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint i   [[thread_position_in_threadgroup]]) {
    threadgroup float xs[TQ_D];
    xs[i] = x[row * TQ_D + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (transpose != 0) {
        for (uint j = 0; j < TQ_D; ++j) acc += R[j * TQ_D + i] * xs[j];
    } else {
        for (uint j = 0; j < TQ_D; ++j) acc += R[i * TQ_D + j] * xs[j];
    }
    x[row * TQ_D + i] = acc;
}

// Igual que tq_rotate_f32, con la salida en f16 (la atención de prefill entrega o en f16).
kernel void tq_rotate_f16(device half*        x         [[buffer(0)]],
                          device const float* R         [[buffer(1)]],
                          constant uint&      transpose [[buffer(2)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint i   [[thread_position_in_threadgroup]]) {
    threadgroup float xs[TQ_D];
    xs[i] = float(x[row * TQ_D + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (transpose != 0) {
        for (uint j = 0; j < TQ_D; ++j) acc += R[j * TQ_D + i] * xs[j];
    } else {
        for (uint j = 0; j < TQ_D; ++j) acc += R[i * TQ_D + j] * xs[j];
    }
    x[row * TQ_D + i] = half(acc);
}

// Un threadgroup de 128 hilos por fila (4 simdgroups de 32). Hilo i: coordenada i de y.
kernel void tq_store_tq4(device const float* src [[buffer(0)]],  // [filas, 128] f32
                         device uchar*       dst [[buffer(1)]],  // [filas, 68]
                         device const float* R   [[buffer(2)]],  // [128, 128]
                         uint row  [[threadgroup_position_in_grid]],
                         uint i    [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[TQ_D];
    threadgroup float part[4];
    threadgroup uchar codes[TQ_D];
    const float v = src[row * TQ_D + i];
    xs[i] = v;
    const float s = simd_sum(v * v);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Norma en f16, como se guarda; los códigos usan esa misma norma (la que decodifica).
    const float norm2 = (part[0] + part[1]) + (part[2] + part[3]);
    const float nh = float(half(sqrt(norm2)));
    float y = 0.0f;
    for (uint j = 0; j < TQ_D; ++j) y += R[i * TQ_D + j] * xs[j];
    const float u = (nh == 0.0f) ? 0.0f : precise::divide(y, nh);
    uchar c = 0;
    for (uint k = 0; k < 15; ++k) c += (u > TQ_MID[k]) ? 1 : 0;
    codes[i] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uchar* out = dst + row * TQ_ROW;
    if (i < 64) out[i] = uchar(codes[2 * i] | (codes[2 * i + 1] << 4));
    if (i == 0) *((device half*)(out + 64)) = half(nh);
    if (i == 1) { out[66] = 0; out[67] = 0; }
}
