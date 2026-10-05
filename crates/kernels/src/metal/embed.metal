// Embedding desde una tabla q8_0 [vocab, H]: out[t, :] = decuantizar(fila ids[t]).
#include <metal_stdlib>
using namespace metal;

struct block_q8_0 { half d; char qs[32]; };

kernel void embed_q8_0(device const block_q8_0* table [[buffer(0)]],
                       device const uint*       ids   [[buffer(1)]],
                       device float*            out   [[buffer(2)]],
                       constant uint&           h     [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // gid.x: columna, gid.y: token.
    uint col = gid.x, t = gid.y;
    if (col >= h) return;
    device const block_q8_0* row = table + ids[t] * (h / 32);
    block_q8_0 b = row[col / 32];
    out[t * h + col] = float(b.d) * float(b.qs[col % 32]);
}
