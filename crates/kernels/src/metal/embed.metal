// Embedding desde una tabla q8_0 o q6_0 (ADR 0012) [vocab, H]: out[t, :] = decuantizar(fila ids[t]).
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

// q6_0 (ADR 0012): d f16, ql[16] (bits 0-3 de j y j+16), qh[8] (bits 4-5 de j, j+8, j+16, j+24).
struct block_q6_0 { half d; uchar ql[16]; uchar qh[8]; };

kernel void embed_q6_0(device const block_q6_0* table [[buffer(0)]],
                       device const uint*       ids   [[buffer(1)]],
                       device float*            out   [[buffer(2)]],
                       constant uint&           h     [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x, t = gid.y;
    if (col >= h) return;
    device const block_q6_0& b = table[ids[t] * (h / 32) + col / 32];
    uint i = col % 32;
    uint lo = (i < 16) ? (b.ql[i] & 0x0F) : (b.ql[i - 16] >> 4);
    uint hi = (b.qh[i % 8] >> (2 * (i / 8))) & 3;
    out[t * h + col] = float(b.d) * float(int(lo | (hi << 4)) - 32);
}
