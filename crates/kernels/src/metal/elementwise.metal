// Operaciones elemento a elemento. Referencia CPU: brasa_kernels::reference.
#include <metal_stdlib>
using namespace metal;

// out[i] = a[i] + b[i]
kernel void add_f32(device const float* a   [[buffer(0)]],
                    device const float* b   [[buffer(1)]],
                    device float*       out [[buffer(2)]],
                    constant uint&      n   [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// SwiGLU: out[i] = silu(gate[i]) * up[i], con silu(x) = x / (1 + e^-x).
kernel void swiglu_f32(device const float* gate [[buffer(0)]],
                       device const float* up   [[buffer(1)]],
                       device float*       out  [[buffer(2)]],
                       constant uint&      n    [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + precise::exp(-g))) * up[i];
    }
}

// SwiGLU con salida en f16 para el GEMM de prefill (ADR 0030): swiglu_f32 redondeado.
kernel void swiglu_f16(device const float* gate [[buffer(0)]],
                       device const float* up   [[buffer(1)]],
                       device half*        out  [[buffer(2)]],
                       constant uint&      n    [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float g = gate[i];
        out[i] = half((g / (1.0f + precise::exp(-g))) * up[i]);
    }
}
