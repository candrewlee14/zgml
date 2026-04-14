//! KV cache for autoregressive transformer inference.
//!
//! Stores computed key and value tensors from previous positions,
//! so each generation step only processes the new token (seq_len=1)
//! instead of reprocessing the entire sequence.
//!
//! Usage:
//! ```
//! var cache = try KVCache(f32).init(alloc, n_layers, n_heads, d_head, max_seq);
//! defer cache.deinit();
//!
//! // Each step: append new K/V, get full cached K/V for attention
//! cache.append(layer, head, new_k_data, new_v_data);
//! const full_k = cache.keys(layer, head);   // [d_head, pos+1]
//! const full_v = cache.values(layer, head);  // [d_head, pos+1]
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;

pub fn KVCache(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        n_layers: usize,
        n_heads: usize,
        d_head: usize,
        max_seq: usize,
        pos: usize, // current position (number of cached tokens)

        // Storage: [n_layers][n_heads][d_head * max_seq] for K and V
        k_data: []T,
        v_data: []T,

        pub fn init(alloc: Alloc, n_layers: usize, n_heads: usize, d_head: usize, max_seq: usize) !Self {
            const total = n_layers * n_heads * d_head * max_seq;
            const k = try alloc.alloc(T, total);
            @memset(k, 0);
            const v = try alloc.alloc(T, total);
            @memset(v, 0);
            return .{
                .alloc = alloc,
                .n_layers = n_layers,
                .n_heads = n_heads,
                .d_head = d_head,
                .max_seq = max_seq,
                .pos = 0,
                .k_data = k,
                .v_data = v,
            };
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.k_data);
            self.alloc.free(self.v_data);
        }

        pub fn reset(self: *Self) void {
            self.pos = 0;
        }

        fn sliceOffset(self: *const Self, layer: usize, head: usize) usize {
            return (layer * self.n_heads + head) * self.d_head * self.max_seq;
        }

        /// Append K/V for one head at the current position.
        /// k_new and v_new are [d_head] slices.
        pub fn append(self: *Self, layer: usize, head: usize, k_new: []const T, v_new: []const T) void {
            std.debug.assert(k_new.len == self.d_head);
            std.debug.assert(v_new.len == self.d_head);
            std.debug.assert(self.pos < self.max_seq);

            const base = self.sliceOffset(layer, head);
            // Storage layout: [d_head, max_seq] row-major
            // Position `pos` occupies column `pos`: data[d * max_seq + pos]
            for (0..self.d_head) |d| {
                self.k_data[base + d * self.max_seq + self.pos] = k_new[d];
                self.v_data[base + d * self.max_seq + self.pos] = v_new[d];
            }
        }

        /// Advance position counter. Call after all layers have appended for this position.
        pub fn advance(self: *Self) void {
            self.pos += 1;
        }

        /// Get cached keys for a head: [d_head, pos] slice.
        /// Returns the raw data pointer and the current sequence length.
        pub fn keysSlice(self: *const Self, layer: usize, head: usize) []const T {
            const base = self.sliceOffset(layer, head);
            return self.k_data[base .. base + self.d_head * self.max_seq];
        }

        pub fn valuesSlice(self: *const Self, layer: usize, head: usize) []const T {
            const base = self.sliceOffset(layer, head);
            return self.v_data[base .. base + self.d_head * self.max_seq];
        }

        pub fn seqLen(self: *const Self) usize {
            return self.pos;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "kv_cache - append and retrieve" {
    const alloc = std.testing.allocator;

    var cache = try KVCache(f32).init(alloc, 1, 1, 2, 4);
    defer cache.deinit();

    // Append position 0
    cache.append(0, 0, &.{ 1.0, 2.0 }, &.{ 3.0, 4.0 });
    cache.advance();
    try std.testing.expectEqual(@as(usize, 1), cache.seqLen());

    // Append position 1
    cache.append(0, 0, &.{ 5.0, 6.0 }, &.{ 7.0, 8.0 });
    cache.advance();
    try std.testing.expectEqual(@as(usize, 2), cache.seqLen());

    // Check stored values (layout: [d_head, max_seq])
    const k = cache.keysSlice(0, 0);
    // d=0: k[0*4+0]=1.0, k[0*4+1]=5.0
    // d=1: k[1*4+0]=2.0, k[1*4+1]=6.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), k[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), k[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), k[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), k[5], 1e-6);
}

test "kv_cache - reset" {
    const alloc = std.testing.allocator;
    var cache = try KVCache(f32).init(alloc, 2, 2, 4, 8);
    defer cache.deinit();

    cache.append(0, 0, &.{ 1, 2, 3, 4 }, &.{ 5, 6, 7, 8 });
    cache.advance();
    try std.testing.expectEqual(@as(usize, 1), cache.seqLen());

    cache.reset();
    try std.testing.expectEqual(@as(usize, 0), cache.seqLen());
}
