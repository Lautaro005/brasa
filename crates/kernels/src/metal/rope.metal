// RoPE estilo NeoX (rotate_half), en el lugar, sobre x: [T, heads, D].
// cos/sin vienen de una tabla [max_pos, D/2] calculada en CPU en f64 al cargar el modelo
// (calcular pos·inv_freq en f32 pierde precisión en posiciones grandes).
// Para i < D/2: x'[i] = x[i]·cos - x[i+D/2]·sin ; x'[i+D/2] = x[i+D/2]·cos + x[i]·sin.
#include <metal_stdlib>
using namespace metal;

kernel void rope_neox_f32(device float*       x     [[buffer(0)]],
                          device const float* cos_t [[buffer(1)]],
                          device const float* sin_t [[buffer(2)]],
                          constant uint&      heads [[buffer(3)]],
                          constant uint&      dim   [[buffer(4)]],
                          constant uint&      pos0  [[buffer(5)]],
                          uint3 gid [[thread_position_in_grid]]) {
    // gid.x: índice i < D/2, gid.y: cabeza, gid.z: token.
    uint half_d = dim / 2;
    uint i = gid.x, h = gid.y, t = gid.z;
    if (i >= half_d || h >= heads) return;
    device float* v = x + (t * heads + h) * dim;
    uint p = pos0 + t;
    float c = cos_t[p * half_d + i];
    float s = sin_t[p * half_d + i];
    float a = v[i];
    float b = v[i + half_d];
    v[i] = a * c - b * s;
    v[i + half_d] = b * c + a * s;
}
