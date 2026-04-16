//! Weight-only int8 quantization for inference.
//!
//! Stores weights as int8 with per-block f32 scales. During matmul,
//! int8 weights are dequantized on the fly — 4x memory reduction with
//! minimal accuracy loss for inference.
//!
//! ```
//! var qw = try QuantizedWeight(f32).fromTensor(allocator, weight_tensor, 32);
//! defer qw.deinit(allocator);
//! qw.matmul(input_data, output_data, M, N, K);
//! ```

const std = @import("std");
const c = @cImport(@cInclude("pthread.h"));

/// Block size for quantization scales. Each block of `block_size` int8 values
/// shares one f32 scale factor.
pub const default_block_size: usize = 32;

/// Persistent GEMV thread pool using pthreads condition variables.
/// Workers sleep on a condvar between dispatches — near-zero wake latency.
/// Threads are spawned lazily on first dispatch so the pool can be
/// returned by value from init.
pub fn GemvPool(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const max_workers = 16;

        const Task = struct {
            t_d: [*]const i8,
            t_s: [*]const T,
            inp_q: [*]const i8,
            inp_scales: [*]const T,
            dst: [*]T,
            n_start: usize,
            n_end: usize,
            K: usize,
            bs: usize,
        };

        const WorkerCtx = struct {
            pool: *Self,
            id: usize,
        };

        n_workers: usize,
        alloc: std.mem.Allocator,
        spawned: bool = false,
        threads: [max_workers]c.pthread_t = undefined,
        ctxs: ?[]WorkerCtx = null,
        tasks: [max_workers]Task = undefined,
        mutex: c.pthread_mutex_t = undefined,
        work_ready: c.pthread_cond_t = undefined,
        work_done: c.pthread_cond_t = undefined,
        generation: usize = 0,
        worker_gens: [max_workers]usize = [_]usize{0} ** max_workers,
        pending: usize = 0,
        shutdown: bool = false,

        pub fn init(alloc: std.mem.Allocator, n_workers: usize) !Self {
            return .{ .n_workers = @min(n_workers, max_workers), .alloc = alloc };
        }

        fn ensureSpawned(self: *Self) void {
            if (self.spawned) return;
            _ = c.pthread_mutex_init(&self.mutex, null);
            _ = c.pthread_cond_init(&self.work_ready, null);
            _ = c.pthread_cond_init(&self.work_done, null);

            const ctxs = self.alloc.alloc(WorkerCtx, self.n_workers) catch {
                self.n_workers = 1;
                self.spawned = true;
                return;
            };
            self.ctxs = ctxs;

            for (1..self.n_workers) |i| {
                ctxs[i] = .{ .pool = self, .id = i };
                if (c.pthread_create(&self.threads[i], null, &workerEntry, @ptrCast(&ctxs[i])) != 0) {
                    self.n_workers = i;
                    break;
                }
            }
            self.spawned = true;
        }

        pub fn deinit(self: *Self, _: std.mem.Allocator) void {
            if (!self.spawned) return;
            _ = c.pthread_mutex_lock(&self.mutex);
            self.shutdown = true;
            _ = c.pthread_cond_broadcast(&self.work_ready);
            _ = c.pthread_mutex_unlock(&self.mutex);
            for (1..self.n_workers) |i| _ = c.pthread_join(self.threads[i], null);
            _ = c.pthread_cond_destroy(&self.work_done);
            _ = c.pthread_cond_destroy(&self.work_ready);
            _ = c.pthread_mutex_destroy(&self.mutex);
            if (self.ctxs) |buf| self.alloc.free(buf);
        }

        fn workerEntry(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
            const ctx: *const WorkerCtx = @ptrCast(@alignCast(arg));
            const id = ctx.id;
            const pool = ctx.pool;
            var my_gen: usize = 0;

            while (true) {
                _ = c.pthread_mutex_lock(&pool.mutex);
                while (pool.worker_gens[id] == my_gen and !pool.shutdown)
                    _ = c.pthread_cond_wait(&pool.work_ready, &pool.mutex);
                if (pool.shutdown) {
                    _ = c.pthread_mutex_unlock(&pool.mutex);
                    return null;
                }
                const task = pool.tasks[id];
                my_gen = pool.worker_gens[id];
                _ = c.pthread_mutex_unlock(&pool.mutex);

                const bpr = (task.K + task.bs - 1) / task.bs;
                QuantizedWeight(T).gemvRange(
                    task.t_d[0 .. task.n_end * task.K],
                    task.t_s[0 .. task.n_end * bpr],
                    task.inp_q[0..task.K],
                    task.inp_scales[0..bpr],
                    task.dst[0..task.n_end],
                    task.n_start, task.n_end, task.K, task.bs,
                );

                _ = c.pthread_mutex_lock(&pool.mutex);
                pool.pending -= 1;
                if (pool.pending == 0) _ = c.pthread_cond_signal(&pool.work_done);
                _ = c.pthread_mutex_unlock(&pool.mutex);
            }
        }

        pub fn dispatch(
            self: *Self,
            qw: *const QuantizedWeight(T),
            inp_q: [*]const i8,
            inp_scales: [*]const T,
            dst: []T,
            N: usize,
            K: usize,
        ) void {
            const bs = qw.block_size;
            const t_d = qw.t_data.?;
            const t_s = qw.t_scales.?;
            const blocks_per_row = (K + bs - 1) / bs;

            // Only thread when total work exceeds threshold.
            const min_work_per_thread: usize = 1024 * 1024;
            const useful = @max(1, (N * K) / min_work_per_thread);
            const n_active = @min(useful, self.n_workers);

            if (n_active <= 1) {
                QuantizedWeight(T).gemvRange(
                    t_d, t_s, inp_q[0..K], inp_scales[0..blocks_per_row],
                    dst, 0, N, K, bs,
                );
                return;
            }

            self.ensureSpawned();

            const chunk = (((N + n_active - 1) / n_active) + 3) & ~@as(usize, 3);

            self.generation +%= 1;
            _ = c.pthread_mutex_lock(&self.mutex);
            var n_dispatched: usize = 0;
            var n_start: usize = chunk;
            for (1..n_active) |i| {
                if (n_start >= N) break;
                self.tasks[i] = .{
                    .t_d = t_d.ptr, .t_s = t_s.ptr,
                    .inp_q = inp_q, .inp_scales = inp_scales,
                    .dst = dst.ptr, .n_start = n_start,
                    .n_end = @min(n_start + chunk, N),
                    .K = K, .bs = bs,
                };
                self.worker_gens[i] = self.generation;
                n_dispatched += 1;
                n_start += chunk;
            }
            self.pending = n_dispatched;
            _ = c.pthread_cond_broadcast(&self.work_ready);
            _ = c.pthread_mutex_unlock(&self.mutex);

            QuantizedWeight(T).gemvRange(
                t_d, t_s, inp_q[0..K], inp_scales[0..blocks_per_row],
                dst, 0, @min(chunk, N), K, bs,
            );

            _ = c.pthread_mutex_lock(&self.mutex);
            while (self.pending > 0)
                _ = c.pthread_cond_wait(&self.work_done, &self.mutex);
            _ = c.pthread_mutex_unlock(&self.mutex);
        }
    };
}

pub fn QuantizedWeight(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []const i8,
        scales: []const T,
        rows: usize, // K (inner dim)
        cols: usize, // N (output dim)
        block_size: usize,
        // N-major transposed data for GEMV: [N, K] layout with K-aligned blocks.
        // Each output row n has ceil(K/block_size) blocks.
        t_data: ?[]const i8 = null,
        t_scales: ?[]const T = null,

        /// Quantize a f32 weight matrix to int8 with per-block scaling.
        /// Weight layout: [K, N] (row-major, K rows, N cols).
        pub fn fromSlice(alloc: std.mem.Allocator, weights: []const T, rows: usize, cols: usize, block_size: usize) !Self {
            std.debug.assert(weights.len == rows * cols);
            const n_elems = rows * cols;
            const n_blocks = (n_elems + block_size - 1) / block_size;

            const data = try alloc.alloc(i8, n_elems);
            errdefer alloc.free(data);
            const scales = try alloc.alloc(T, n_blocks);
            errdefer alloc.free(scales);

            // Quantize each block
            for (0..n_blocks) |b| {
                const start = b * block_size;
                const end = @min(start + block_size, n_elems);
                const block = weights[start..end];

                // Find max absolute value in block
                var max_abs: T = 0;
                for (block) |v| {
                    const a = @abs(v);
                    if (a > max_abs) max_abs = a;
                }

                const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
                scales[b] = scale;
                const inv_scale = if (max_abs > 0) 127.0 / max_abs else 0.0;

                for (block, start..) |v, j| {
                    const q = v * inv_scale;
                    data[j] = @intFromFloat(std.math.clamp(q, -127.0, 127.0));
                }
            }

            return .{
                .data = data,
                .scales = scales,
                .rows = rows,
                .cols = cols,
                .block_size = block_size,
            };
        }

        /// Quantize from a col-major 2D Tensor.
        ///
        /// Col-major [ne0, ne1] has the same flat layout as row-major [ne1, ne0].
        /// The quantized weight stores [K=ne1, N=ne0] for matmul dispatch.
        pub fn fromTensor(alloc: std.mem.Allocator, tensor: anytype, block_size: usize) !Self {
            return fromSlice(alloc, tensor.data, tensor.ne[1], tensor.ne[0], block_size);
        }

        pub fn deinit(self: Self, alloc: std.mem.Allocator) void {
            alloc.free(self.data);
            alloc.free(self.scales);
            if (self.t_data) |d| alloc.free(d);
            if (self.t_scales) |s| alloc.free(s);
        }

        /// Build transposed [N, K] layout for fast GEMV (M=1).
        pub fn prepareTransposed(self: *Self, alloc: std.mem.Allocator) !void {
            const K = self.rows;
            const N = self.cols;
            const bs = self.block_size;
            const blocks_per_row = (K + bs - 1) / bs;

            const t_data = try alloc.alloc(i8, N * K);
            errdefer alloc.free(t_data);
            const t_scales = try alloc.alloc(T, N * blocks_per_row);
            errdefer alloc.free(t_scales);

            for (0..N) |n| {
                for (0..blocks_per_row) |b| {
                    const k_start = b * bs;
                    const k_end = @min(k_start + bs, K);

                    var max_abs: T = 0;
                    for (k_start..k_end) |k| {
                        const orig_flat = k * N + n;
                        const orig_block = orig_flat / bs;
                        const val = @as(T, @floatFromInt(self.data[orig_flat])) * self.scales[orig_block];
                        const a = @abs(val);
                        if (a > max_abs) max_abs = a;
                    }

                    const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
                    const inv_scale = if (max_abs > 0) 127.0 / max_abs else 0.0;
                    t_scales[n * blocks_per_row + b] = scale;

                    for (k_start..k_end) |k| {
                        const orig_flat = k * N + n;
                        const orig_block = orig_flat / bs;
                        const val = @as(T, @floatFromInt(self.data[orig_flat])) * self.scales[orig_block];
                        const q = val * inv_scale;
                        t_data[n * K + k] = @intFromFloat(std.math.clamp(q, -127.0, 127.0));
                    }
                }
            }

            if (self.t_data) |d| alloc.free(d);
            if (self.t_scales) |s| alloc.free(s);
            self.t_data = t_data;
            self.t_scales = t_scales;
        }

        /// Quantize f32 input to int8 per-block. Used by GEMV before integer dot products.
        pub fn quantizeInput(input: []const T, K: usize, bs: usize, inp_q: []i8, inp_scales: []T) void {
            const blocks_per_row = (K + bs - 1) / bs;
            for (0..blocks_per_row) |b| {
                const k_start = b * bs;
                const k_end = @min(k_start + bs, K);
                const block = input[k_start..k_end];

                var max_abs: T = 0;
                for (block) |v| {
                    const a = @abs(v);
                    if (a > max_abs) max_abs = a;
                }

                const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
                const inv_scale = if (max_abs > 0) 127.0 / max_abs else 0.0;
                inp_scales[b] = scale;

                for (block, k_start..) |v, j| {
                    inp_q[j] = @intFromFloat(std.math.clamp(v * inv_scale, -127.0, 127.0));
                }
            }
        }

        const I8x16 = @Vector(16, i8);
        const I32x4 = @Vector(4, i32);

        /// ARM SDOT: acc += dot4(a, b) per lane. 4×i8→i32 in one cycle.
        inline fn armSdot(acc: I32x4, a: I8x16, b: I8x16) I32x4 {
            return asm ("sdot %[acc].4s, %[a].16b, %[b].16b"
                : [acc] "=w" (-> I32x4),
                : [_] "0" (acc),
                  [a] "w" (a),
                  [b] "w" (b),
            );
        }

        /// Compute GEMV for a range of output rows [n_start, n_end).
        /// Uses ARM SDOT for the inner loop (4×i8→i32 dot product per cycle).
        pub fn gemvRange(
            t_d: []const i8,
            t_s: []const T,
            inp_q: []const i8,
            inp_scales: []const T,
            dst: []T,
            n_start: usize,
            n_end: usize,
            K: usize,
            bs: usize,
        ) void {
            const blocks_per_row = (K + bs - 1) / bs;
            const vec_len: usize = 16;
            const n_unroll: usize = 4;

            var n: usize = n_start;
            while (n + n_unroll <= n_end) : (n += n_unroll) {
                var accs: [n_unroll]T = .{0} ** n_unroll;

                for (0..blocks_per_row) |b| {
                    const k_start = b * bs;
                    const k_end = @min(k_start + bs, K);
                    const block_len = k_end - k_start;

                    var combined_scales: [n_unroll]T = undefined;
                    inline for (0..n_unroll) |ni| {
                        combined_scales[ni] = inp_scales[b] * t_s[(n + ni) * blocks_per_row + b];
                    }

                    var sdot_accs: [n_unroll]I32x4 = .{@as(I32x4, @splat(0))} ** n_unroll;
                    var j: usize = 0;
                    while (j + vec_len <= block_len) : (j += vec_len) {
                        const inp_vec: I8x16 = inp_q[k_start + j ..][0..vec_len].*;
                        inline for (0..n_unroll) |ni| {
                            const w_vec: I8x16 = t_d[(n + ni) * K + k_start + j ..][0..vec_len].*;
                            sdot_accs[ni] = armSdot(sdot_accs[ni], inp_vec, w_vec);
                        }
                    }

                    var int_accs: [n_unroll]i32 = undefined;
                    inline for (0..n_unroll) |ni| {
                        int_accs[ni] = @reduce(.Add, sdot_accs[ni]);
                    }
                    while (j < block_len) : (j += 1) {
                        inline for (0..n_unroll) |ni| {
                            int_accs[ni] += @as(i32, inp_q[k_start + j]) * @as(i32, t_d[(n + ni) * K + k_start + j]);
                        }
                    }

                    inline for (0..n_unroll) |ni| {
                        accs[ni] += @as(T, @floatFromInt(int_accs[ni])) * combined_scales[ni];
                    }
                }

                inline for (0..n_unroll) |ni| {
                    dst[n + ni] = accs[ni];
                }
            }

            while (n < n_end) : (n += 1) {
                var acc: T = 0;
                for (0..blocks_per_row) |b| {
                    const k_start = b * bs;
                    const k_end = @min(k_start + bs, K);
                    const block_len = k_end - k_start;
                    const combined_scale = inp_scales[b] * t_s[n * blocks_per_row + b];

                    var sdot_acc: I32x4 = @splat(0);
                    var j: usize = 0;
                    while (j + vec_len <= block_len) : (j += vec_len) {
                        const inp_vec: I8x16 = inp_q[k_start + j ..][0..vec_len].*;
                        const w_vec: I8x16 = t_d[n * K + k_start + j ..][0..vec_len].*;
                        sdot_acc = armSdot(sdot_acc, inp_vec, w_vec);
                    }
                    var int_acc: i32 = @reduce(.Add, sdot_acc);
                    while (j < block_len) : (j += 1) {
                        int_acc += @as(i32, inp_q[k_start + j]) * @as(i32, t_d[n * K + k_start + j]);
                    }
                    acc += @as(T, @floatFromInt(int_acc)) * combined_scale;
                }
                dst[n] = acc;
            }
        }

        /// Single-threaded GEMV for M=1. For threaded dispatch, use GemvPool.
        pub fn gemv(self: *const Self, input: []const T, dst: []T, N: usize, K: usize) void {
            std.debug.assert(self.t_data != null);
            std.debug.assert(self.t_scales != null);
            std.debug.assert(self.rows == K);
            std.debug.assert(self.cols == N);

            const bs = self.block_size;
            const blocks_per_row = (K + bs - 1) / bs;

            var inp_q_buf: [16384]i8 = undefined;
            var inp_scales_buf: [512]T = undefined;
            const inp_q = inp_q_buf[0..K];
            const inp_scales = inp_scales_buf[0..blocks_per_row];
            quantizeInput(input, K, bs, inp_q, inp_scales);

            gemvRange(self.t_data.?, self.t_scales.?, inp_q, inp_scales, dst, 0, N, K, bs);
        }

        /// Dequantize a single element.
        fn dequant(self: *const Self, idx: usize) T {
            const block = idx / self.block_size;
            return @as(T, @floatFromInt(self.data[idx])) * self.scales[block];
        }

        /// Quantized matmul: dst[m,n] = sum_k(input[m,k] * dequant(weight[k,n]))
        /// input: [M, K] row-major f32
        /// weight (self): [K, N] row-major int8
        /// dst: [M, N] row-major f32
        ///
        /// Loop order M→K→N for sequential weight reads.  The inner N loop
        /// processes one quantization block at a time (hoisted scale) with
        /// explicit SIMD.
        pub fn matmul(self: *const Self, input: []const T, dst: []T, M: usize, N: usize, K: usize) void {
            std.debug.assert(self.rows == K);
            std.debug.assert(self.cols == N);
            std.debug.assert(input.len >= M * K);
            std.debug.assert(dst.len >= M * N);

            const bs = self.block_size;
            const w_data = self.data;
            const scales = self.scales;

            // Zero destination.
            @memset(dst[0 .. M * N], 0);

            const vec_len = comptime @min(8, default_block_size);
            const Vec = @Vector(vec_len, T);
            const IVec = @Vector(vec_len, i32);
            const I8Vec = @Vector(vec_len, i8);

            const k_unroll: usize = 4;

            for (0..M) |m| {
                const dst_row = dst[m * N ..][0..N];
                const inp_row = input[m * K ..][0..K];

                // --- K-unrolled loop: process k_unroll K values at a time ---
                var k: usize = 0;
                while (k + k_unroll <= K) : (k += k_unroll) {
                    // Load k_unroll input values.
                    var inp_vals: [k_unroll]T = undefined;
                    var w_bases: [k_unroll]usize = undefined;
                    inline for (0..k_unroll) |ki| {
                        inp_vals[ki] = inp_row[k + ki];
                        w_bases[ki] = (k + ki) * N;
                    }

                    // Walk N in quantization-block-aligned chunks.
                    // All k_unroll weight rows share the same N layout, but each
                    // row has its own quant blocks (different scales).
                    var n: usize = 0;
                    while (n < N) {
                        // Determine chunk size from first row's block alignment
                        // (all rows have the same column layout).
                        const flat0 = w_bases[0] + n;
                        const block_rem = bs - (flat0 % bs);
                        const chunk = @min(block_rem, N - n);

                        // Precompute combined scale*input vectors for each k.
                        var combined: [k_unroll]Vec = undefined;
                        inline for (0..k_unroll) |ki| {
                            const flat_ki = w_bases[ki] + n;
                            const scale_ki = scales[flat_ki / bs];
                            combined[ki] = @splat(scale_ki * inp_vals[ki]);
                        }

                        // SIMD inner loop: load dst once, accumulate k_unroll FMAs, store once.
                        var j: usize = 0;
                        while (j + vec_len <= chunk) : (j += vec_len) {
                            var d_vec: Vec = dst_row[n + j ..][0..vec_len].*;
                            inline for (0..k_unroll) |ki| {
                                const w_vec: I8Vec = w_data[w_bases[ki] + n + j ..][0..vec_len].*;
                                const f_vec: Vec = @floatFromInt(@as(IVec, w_vec));
                                d_vec += f_vec * combined[ki];
                            }
                            dst_row[n + j ..][0..vec_len].* = d_vec;
                        }
                        // Scalar tail for N.
                        while (j < chunk) : (j += 1) {
                            inline for (0..k_unroll) |ki| {
                                const flat_ki = w_bases[ki] + n + j;
                                dst_row[n + j] += inp_vals[ki] * @as(T, @floatFromInt(w_data[flat_ki])) * scales[flat_ki / bs];
                            }
                        }
                        n += chunk;
                    }
                }

                // --- Scalar tail for remaining K values ---
                while (k < K) : (k += 1) {
                    const inp_val = inp_row[k];
                    const w_base = k * N;

                    var n: usize = 0;
                    while (n < N) {
                        const flat = w_base + n;
                        const scale = scales[flat / bs];
                        const combined_s: Vec = @splat(scale * inp_val);
                        const block_rem = bs - (flat % bs);
                        const chunk = @min(block_rem, N - n);

                        var j: usize = 0;
                        while (j + vec_len <= chunk) : (j += vec_len) {
                            const w_vec: I8Vec = w_data[flat + j ..][0..vec_len].*;
                            const f_vec: Vec = @floatFromInt(@as(IVec, w_vec));
                            const d_vec: Vec = dst_row[n + j ..][0..vec_len].*;
                            dst_row[n + j ..][0..vec_len].* = d_vec + f_vec * combined_s;
                        }
                        while (j < chunk) : (j += 1) {
                            dst_row[n + j] += inp_val * @as(T, @floatFromInt(w_data[flat + j])) * scale;
                        }
                        n += chunk;
                    }
                }
            }
        }

        /// Quantized matmul with bias addition.
        pub fn matmulBias(self: *const Self, input: []const T, bias: []const T, dst: []T, M: usize, N: usize, K: usize) void {
            self.matmul(input, dst, M, N, K);
            // Add bias (broadcast over M rows)
            for (0..M) |m| {
                for (0..N) |n| {
                    dst[m * N + n] += bias[n];
                }
            }
        }

        /// Compute quantization error (RMSE) vs original weights.
        pub fn quantizationError(self: *const Self, original: []const T) T {
            var sum_sq: T = 0;
            for (original, 0..) |orig, j| {
                const diff = orig - self.dequant(j);
                sum_sq += diff * diff;
            }
            return @sqrt(sum_sq / @as(T, @floatFromInt(original.len)));
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "quantize and dequantize preserves values approximately" {
    const alloc = testing.allocator;
    const weights = [_]f32{ 1.0, -0.5, 0.25, -1.0, 0.0, 0.75, -0.3, 0.9 };

    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 2, 4, 4);
    defer qw.deinit(alloc);

    // Quantization error should be small
    const err = qw.quantizationError(&weights);
    try testing.expect(err < 0.01);
}

test "quantized matmul matches float matmul approximately" {
    const alloc = testing.allocator;

    // Weight: 3x2 (K=3, N=2)
    const weights = [_]f32{
        1.0,  0.5,
        -0.5, 1.0,
        0.25, -0.25,
    };
    // Input: 2x3 (M=2, K=3)
    const input = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };

    // Reference float matmul
    var ref_output: [4]f32 = undefined;
    for (0..2) |m| {
        for (0..2) |n| {
            var acc: f32 = 0;
            for (0..3) |k| {
                acc += input[m * 3 + k] * weights[k * 2 + n];
            }
            ref_output[m * 2 + n] = acc;
        }
    }

    // Quantized matmul
    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 3, 2, 32);
    defer qw.deinit(alloc);

    var q_output: [4]f32 = undefined;
    qw.matmul(&input, &q_output, 2, 2, 3);

    // Should be close (within quantization error)
    for (ref_output, q_output) |r, q| {
        try testing.expectApproxEqAbs(r, q, 0.1);
    }
}

test "gemv matches matmul for M=1" {
    const alloc = testing.allocator;

    // Weight: 4x8 (K=4, N=8)
    const weights = [_]f32{
        1.0,  0.5,  -0.3, 0.8,  -1.0, 0.2,  0.7,  -0.4,
        -0.5, 1.0,  0.6,  -0.9, 0.3,  -0.7, 0.1,  0.5,
        0.25, -0.25, 1.0, 0.4,  -0.6, 0.9,  -0.2, 0.3,
        0.7,  -0.8, 0.15, 1.0,  0.5,  -0.3, 0.6,  -0.1,
    };
    const input = [_]f32{ 1.0, 2.0, -0.5, 0.3 };

    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 4, 8, 4);
    defer qw.deinit(alloc);
    try qw.prepareTransposed(alloc);

    // matmul path (M=1)
    var matmul_out: [8]f32 = undefined;
    qw.matmul(&input, &matmul_out, 1, 8, 4);

    // gemv path
    var gemv_out: [8]f32 = undefined;
    qw.gemv(&input, &gemv_out, 8, 4);

    // Both should produce the same result (same quantized weights, different layout).
    for (matmul_out, gemv_out) |m, g| {
        try testing.expectApproxEqAbs(m, g, 0.15);
    }

    // Also check against float reference.
    for (0..8) |n| {
        var acc: f32 = 0;
        for (0..4) |k| {
            acc += input[k] * weights[k * 8 + n];
        }
        try testing.expectApproxEqAbs(acc, gemv_out[n], 0.15);
    }
}

test "quantized matmul with bias" {
    const alloc = testing.allocator;
    const weights = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // 2x2 identity
    const bias = [_]f32{ 0.5, -0.5 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // 2x2

    var qw = try QuantizedWeight(f32).fromSlice(alloc, &weights, 2, 2, 32);
    defer qw.deinit(alloc);

    var output: [4]f32 = undefined;
    qw.matmulBias(&input, &bias, &output, 2, 2, 2);

    // Identity + bias: [1.5, 1.5, 3.5, 3.5]
    try testing.expectApproxEqAbs(@as(f32, 1.5), output[0], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 1.5), output[1], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 3.5), output[2], 0.1);
    try testing.expectApproxEqAbs(@as(f32, 3.5), output[3], 0.1);
}

test "block size affects quantization error" {
    const alloc = testing.allocator;
    // Random-ish weights with varying magnitudes
    const weights = [_]f32{ 0.1, 10.0, -0.01, 5.0, 0.5, -8.0, 0.001, 3.0 };

    // Smaller blocks = better accuracy (each block has its own scale)
    var qw_large = try QuantizedWeight(f32).fromSlice(alloc, &weights, 1, 8, 8);
    defer qw_large.deinit(alloc);
    var qw_small = try QuantizedWeight(f32).fromSlice(alloc, &weights, 1, 8, 2);
    defer qw_small.deinit(alloc);

    const err_large = qw_large.quantizationError(&weights);
    const err_small = qw_small.quantizationError(&weights);

    // Smaller blocks should have less error
    try testing.expect(err_small <= err_large);
}
