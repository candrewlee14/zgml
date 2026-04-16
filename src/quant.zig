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
// Quantized KV cache (Q8_0 block layout)
// ---------------------------------------------------------------------------

/// Column-major quantized KV cache.
///
/// `q_data`: [d_head * n_cols] int8. Column `c` at `q_data[c*d_head ..][0..d_head]`.
/// `scales`: [blocks_per_col * n_cols] f32. Column `c`'s scales at
///           `scales[c*blocks_per_col ..][0..blocks_per_col]`.
///
/// d_head must be divisible by block_size. Storage per column is
/// `d_head + blocks_per_col * sizeof(T)` bytes — ~3.6× smaller than f32 for
/// d_head=64, block_size=32.
pub fn QuantizedKVCache(comptime T: type) type {
    return struct {
        const Self = @This();

        d_head: usize,
        n_cols: usize,
        block_size: usize,
        blocks_per_col: usize,

        q_data: []i8,
        scales: []T,

        pub fn init(alloc: std.mem.Allocator, d_head: usize, n_cols: usize, block_size: usize) !Self {
            std.debug.assert(d_head % block_size == 0);
            const bpc = d_head / block_size;
            const q_data = try alloc.alloc(i8, d_head * n_cols);
            errdefer alloc.free(q_data);
            const scales = try alloc.alloc(T, bpc * n_cols);
            errdefer alloc.free(scales);
            @memset(q_data, 0);
            @memset(scales, 0);
            return .{
                .d_head = d_head,
                .n_cols = n_cols,
                .block_size = block_size,
                .blocks_per_col = bpc,
                .q_data = q_data,
                .scales = scales,
            };
        }

        pub fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            alloc.free(self.q_data);
            alloc.free(self.scales);
        }

        pub fn clear(self: *Self) void {
            @memset(self.q_data, 0);
            @memset(self.scales, 0);
        }

        /// Quantize `src` f32 [d_head] into column `col_idx`.
        pub fn storeColumn(self: *Self, col_idx: usize, src: []const T) void {
            std.debug.assert(src.len == self.d_head);
            std.debug.assert(col_idx < self.n_cols);
            const q_off = col_idx * self.d_head;
            const s_off = col_idx * self.blocks_per_col;
            QuantizedWeight(T).quantizeInput(
                src,
                self.d_head,
                self.block_size,
                self.q_data[q_off..][0..self.d_head],
                self.scales[s_off..][0..self.blocks_per_col],
            );
        }

        /// Dequantize column `col_idx` into `dst` f32 [d_head].
        pub fn dequantColumn(self: *const Self, col_idx: usize, dst: []T) void {
            std.debug.assert(dst.len == self.d_head);
            const q_col = self.q_data[col_idx * self.d_head ..][0..self.d_head];
            const sc = self.scales[col_idx * self.blocks_per_col ..][0..self.blocks_per_col];
            for (0..self.blocks_per_col) |b| {
                const base = b * self.block_size;
                const s = sc[b];
                for (0..self.block_size) |i| {
                    dst[base + i] = @as(T, @floatFromInt(q_col[base + i])) * s;
                }
            }
        }

        /// Dot product: `f_vec · dequant(col_idx)`.
        pub fn dotF32(self: *const Self, f_vec: []const T, col_idx: usize) T {
            std.debug.assert(f_vec.len == self.d_head);
            const q_col = self.q_data[col_idx * self.d_head ..][0..self.d_head];
            const sc = self.scales[col_idx * self.blocks_per_col ..][0..self.blocks_per_col];
            return dotI8F32(T, f_vec, q_col, sc, self.block_size, self.blocks_per_col);
        }

        /// SDOT-accelerated dot: pre-quantized `q_i8` (with `q_scales`) ·
        /// dequant(col_idx). Caller must quantize the f32 query once via
        /// `QuantizedWeight(T).quantizeInput`, then call this across many
        /// columns to amortize the quantization.
        pub fn dotI8(
            self: *const Self,
            q_i8: []const i8,
            q_scales: []const T,
            col_idx: usize,
        ) T {
            std.debug.assert(q_i8.len == self.d_head);
            const q_col = self.q_data[col_idx * self.d_head ..][0..self.d_head];
            const sc = self.scales[col_idx * self.blocks_per_col ..][0..self.blocks_per_col];
            return dotI8I8(T, q_i8, q_scales, q_col, sc, self.block_size, self.blocks_per_col);
        }

        /// `acc[i] += w * dequant(col_idx)[i]`.
        pub fn accumF32(self: *const Self, acc: []T, w: T, col_idx: usize) void {
            std.debug.assert(acc.len == self.d_head);
            const q_col = self.q_data[col_idx * self.d_head ..][0..self.d_head];
            const sc = self.scales[col_idx * self.blocks_per_col ..][0..self.blocks_per_col];
            accumI8F32(T, acc, w, q_col, sc, self.block_size, self.blocks_per_col);
        }

        /// Batched accumulate: `acc[i] += sum_{b=0..Bs}(ws[b] * dequant(col_start+b)[i])`.
        /// Keeps `acc` in registers across the Bs-wide inner sum so we pay one
        /// pair of loads/stores on `acc` per tile instead of per column.
        pub fn accumBatchF32(
            self: *const Self,
            comptime Bs: usize,
            acc: []T,
            ws: *const [Bs]T,
            col_start: usize,
        ) void {
            std.debug.assert(acc.len == self.d_head);
            std.debug.assert(col_start + Bs <= self.n_cols);
            accumBatchI8F32(T, Bs, acc, ws, self.q_data, self.scales, col_start, self.d_head, self.block_size, self.blocks_per_col);
        }
    };
}

/// Int8×int8 dot with per-block scales, SDOT-accelerated on aarch64.
/// `q_i8` and `k_i8` are int8 vectors sharing the same block structure.
fn dotI8I8(
    comptime T: type,
    q_i8: []const i8,
    q_scales: []const T,
    k_i8: []const i8,
    k_scales: []const T,
    block_size: usize,
    n_blocks: usize,
) T {
    const I8x16 = @Vector(16, i8);
    const I32x4 = @Vector(4, i32);
    var total: T = 0;
    for (0..n_blocks) |b| {
        const base = b * block_size;
        var sdot_acc: I32x4 = @splat(0);
        var j: usize = 0;
        while (j + 16 <= block_size) : (j += 16) {
            const qv: I8x16 = q_i8[base + j ..][0..16].*;
            const kv: I8x16 = k_i8[base + j ..][0..16].*;
            sdot_acc = asm ("sdot %[acc].4s, %[a].16b, %[b].16b"
                : [acc] "=w" (-> I32x4),
                : [_] "0" (sdot_acc),
                  [a] "w" (qv),
                  [b] "w" (kv),
            );
        }
        var int_sum: i32 = @reduce(.Add, sdot_acc);
        while (j < block_size) : (j += 1) {
            int_sum += @as(i32, q_i8[base + j]) * @as(i32, k_i8[base + j]);
        }
        total += @as(T, @floatFromInt(int_sum)) * q_scales[b] * k_scales[b];
    }
    return total;
}

fn dotI8F32(
    comptime T: type,
    f_vec: []const T,
    q_col: []const i8,
    scales: []const T,
    block_size: usize,
    n_blocks: usize,
) T {
    const V = 8;
    const VecT = @Vector(V, T);
    const VecI = @Vector(V, i32);
    const VecI8 = @Vector(V, i8);
    var total: T = 0;
    for (0..n_blocks) |b| {
        const base = b * block_size;
        var block_sum: VecT = @splat(0);
        var i: usize = 0;
        while (i + V <= block_size) : (i += V) {
            const fv: VecT = f_vec[base + i ..][0..V].*;
            const iv: VecI8 = q_col[base + i ..][0..V].*;
            const qv: VecT = @floatFromInt(@as(VecI, iv));
            block_sum += fv * qv;
        }
        var sub: T = @reduce(.Add, block_sum);
        while (i < block_size) : (i += 1) {
            sub += f_vec[base + i] * @as(T, @floatFromInt(q_col[base + i]));
        }
        total += sub * scales[b];
    }
    return total;
}

fn accumI8F32(
    comptime T: type,
    acc: []T,
    w: T,
    q_col: []const i8,
    scales: []const T,
    block_size: usize,
    n_blocks: usize,
) void {
    const V = 8;
    const VecT = @Vector(V, T);
    const VecI = @Vector(V, i32);
    const VecI8 = @Vector(V, i8);
    for (0..n_blocks) |b| {
        const base = b * block_size;
        const ws_scalar = w * scales[b];
        const ws: VecT = @splat(ws_scalar);
        var i: usize = 0;
        while (i + V <= block_size) : (i += V) {
            const av: VecT = acc[base + i ..][0..V].*;
            const iv: VecI8 = q_col[base + i ..][0..V].*;
            const qv: VecT = @floatFromInt(@as(VecI, iv));
            acc[base + i ..][0..V].* = av + ws * qv;
        }
        while (i < block_size) : (i += 1) {
            acc[base + i] += ws_scalar * @as(T, @floatFromInt(q_col[base + i]));
        }
    }
}

/// Batched V-accumulate: `acc[i] += sum_{b=0..Bs}(ws[b] * dequant(V[col_start+b])[i])`.
/// For each block, pre-scales `ws[b]` by the per-column block scale, then fuses
/// Bs widen+FMAs so `acc` stays in registers across the whole Bs-wide inner sum.
fn accumBatchI8F32(
    comptime T: type,
    comptime Bs: usize,
    acc: []T,
    ws: *const [Bs]T,
    q_data: []const i8,
    scales: []const T,
    col_start: usize,
    d_head: usize,
    block_size: usize,
    n_blocks: usize,
) void {
    const V = 8;
    const VecT = @Vector(V, T);
    const VecI = @Vector(V, i32);
    const VecI8 = @Vector(V, i8);

    for (0..n_blocks) |bl| {
        const block_off = bl * block_size;

        var ws_scaled: [Bs]T = undefined;
        inline for (0..Bs) |b| {
            ws_scaled[b] = ws[b] * scales[(col_start + b) * n_blocks + bl];
        }

        var i: usize = 0;
        while (i + V <= block_size) : (i += V) {
            var sum_v: VecT = acc[block_off + i ..][0..V].*;
            inline for (0..Bs) |b| {
                const iv: VecI8 = q_data[(col_start + b) * d_head + block_off + i ..][0..V].*;
                const qv: VecT = @floatFromInt(@as(VecI, iv));
                const wv: VecT = @splat(ws_scaled[b]);
                sum_v = sum_v + wv * qv;
            }
            acc[block_off + i ..][0..V].* = sum_v;
        }
        while (i < block_size) : (i += 1) {
            var sum_s: T = acc[block_off + i];
            inline for (0..Bs) |b| {
                sum_s += ws_scaled[b] * @as(T, @floatFromInt(q_data[(col_start + b) * d_head + block_off + i]));
            }
            acc[block_off + i] = sum_s;
        }
    }
}

/// Flash attention with quantized K/V cache.
///
/// Output dst (f32) is written in-place. Q is f32. K and V come from
/// `QuantizedKVCache`s that were populated via `storeColumn`. The kernel
/// iterates over columns `[k_col_start, k_col_start + seq_kv)` of `k_cache`
/// and likewise for `v_cache`. Offsets let several heads share one backing
/// cache via contiguous slabs.
///
/// Shapes (column-major):
///   q:    [d_head, seq_q], column stride `q_col_stride`
///   dst:  [d_head, seq_q], column stride `dst_col_stride`
///   mask: optional [seq_kv, seq_q_or_1] with strides (row_stride, col_stride).
///         Passing `mask_col_stride = 0` broadcasts a single mask column.
pub fn attentionQuantized(
    comptime T: type,
    dst: []T,
    dst_col_stride: usize,
    q: []const T,
    q_col_stride: usize,
    d_head: usize,
    seq_q: usize,
    k_cache: *const QuantizedKVCache(T),
    k_col_start: usize,
    v_cache: *const QuantizedKVCache(T),
    v_col_start: usize,
    seq_kv: usize,
    mask: ?[]const T,
    mask_row_stride: usize,
    mask_col_stride: usize,
    scale: T,
) void {
    std.debug.assert(k_cache.d_head == d_head and v_cache.d_head == d_head);
    std.debug.assert(k_col_start + seq_kv <= k_cache.n_cols);
    std.debug.assert(v_col_start + seq_kv <= v_cache.n_cols);

    const max_d_head: usize = 512;
    const max_blocks = max_d_head / 16;
    std.debug.assert(d_head <= max_d_head);
    var acc_buf: [max_d_head]T = undefined;
    const acc = acc_buf[0..d_head];

    // SDOT fast path: pre-quantize each query column to int8 once so the
    // inner dot can go int8×int8 via SDOT instead of widen-and-FMA.
    const use_sdot = @import("builtin").cpu.arch == .aarch64 and
        T == f32 and k_cache.block_size % 16 == 0;
    var q_i8_buf: [max_d_head]i8 = undefined;
    var q_scales_buf: [max_blocks]T = undefined;

    // Flash-attention tile width: amortizes the acc alpha-rescale across Bs
    // columns and lets the batched V-accum keep acc hot in registers.
    const Bs: usize = 8;
    var scores_buf: [Bs]T = undefined;
    var ws_buf: [Bs]T = undefined;

    const neg_inf = -std.math.inf(T);

    for (0..seq_q) |qi| {
        const q_base = qi * q_col_stride;
        const q_col = q[q_base..][0..d_head];
        const mask_base = qi * mask_col_stride;

        const q_i8 = q_i8_buf[0..d_head];
        const q_scales = q_scales_buf[0..k_cache.blocks_per_col];
        if (use_sdot) {
            QuantizedWeight(T).quantizeInput(q_col, d_head, k_cache.block_size, q_i8, q_scales);
        }

        var m_val: T = neg_inf;
        var l: T = 0;
        @memset(acc, 0);

        var s: usize = 0;
        while (s + Bs <= seq_kv) : (s += Bs) {
            // Phase 1: compute Bs scores, find tile max.
            var tile_max: T = neg_inf;
            inline for (0..Bs) |b| {
                const dot = if (use_sdot)
                    k_cache.dotI8(q_i8, q_scales, k_col_start + s + b)
                else
                    k_cache.dotF32(q_col, k_col_start + s + b);
                const mask_add: T = if (mask) |md| md[mask_base + (s + b) * mask_row_stride] else 0;
                const score = dot * scale + mask_add;
                scores_buf[b] = score;
                if (std.math.isFinite(score) and score > tile_max) tile_max = score;
            }

            if (tile_max == neg_inf) continue; // whole tile masked

            // Phase 2: flash-softmax update (one rescale, Bs weights).
            const new_m = if (m_val == neg_inf) tile_max else @max(m_val, tile_max);
            const alpha: T = if (m_val == neg_inf) 0 else @exp(m_val - new_m);

            var tile_l: T = 0;
            inline for (0..Bs) |b| {
                // Masked lanes (-inf) naturally give 0 via exp(-inf) = 0.
                const w = @exp(scores_buf[b] - new_m);
                ws_buf[b] = w;
                tile_l += w;
            }

            if (m_val != neg_inf and alpha != 1) {
                const V = 8;
                const VecT = @Vector(V, T);
                const alpha_v: VecT = @splat(alpha);
                var r: usize = 0;
                while (r + V <= d_head) : (r += V) {
                    const av: VecT = acc[r..][0..V].*;
                    acc[r..][0..V].* = av * alpha_v;
                }
                while (r < d_head) : (r += 1) acc[r] *= alpha;
            }

            // Phase 3: batched V-accum, acc stays in registers across Bs cols.
            v_cache.accumBatchF32(Bs, acc, &ws_buf, v_col_start + s);

            l = l * alpha + tile_l;
            m_val = new_m;
        }

        // Tail: remaining < Bs columns, one at a time.
        while (s < seq_kv) : (s += 1) {
            const dot = if (use_sdot)
                k_cache.dotI8(q_i8, q_scales, k_col_start + s)
            else
                k_cache.dotF32(q_col, k_col_start + s);
            const mask_add: T = if (mask) |md| md[mask_base + s * mask_row_stride] else 0;
            const score = dot * scale + mask_add;
            if (!std.math.isFinite(score)) continue;

            const new_m = @max(m_val, score);
            const alpha: T = if (m_val == neg_inf) 0 else @exp(m_val - new_m);
            const w = @exp(score - new_m);

            if (m_val != neg_inf and alpha != 1) {
                const V = 8;
                const VecT = @Vector(V, T);
                const alpha_v: VecT = @splat(alpha);
                var r: usize = 0;
                while (r + V <= d_head) : (r += V) {
                    const av: VecT = acc[r..][0..V].*;
                    acc[r..][0..V].* = av * alpha_v;
                }
                while (r < d_head) : (r += 1) acc[r] *= alpha;
            }
            v_cache.accumF32(acc, w, v_col_start + s);

            l = l * alpha + w;
            m_val = new_m;
        }

        const inv_l: T = if (l > 0) @as(T, 1) / l else 0;
        const out_base = qi * dst_col_stride;
        const V = 8;
        const VecT = @Vector(V, T);
        const inv_v: VecT = @splat(inv_l);
        var r: usize = 0;
        while (r + V <= d_head) : (r += V) {
            const av: VecT = acc[r..][0..V].*;
            dst[out_base + r ..][0..V].* = av * inv_v;
        }
        while (r < d_head) : (r += 1) dst[out_base + r] = acc[r] * inv_l;
    }
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

// ---------------------------------------------------------------------------
// QuantizedKVCache tests
// ---------------------------------------------------------------------------

fn fillRandF32(buf: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (buf) |*v| v.* = (rng.float(f32) - 0.5) * 2.0;
}

test "QuantizedKVCache - store then dequant roundtrip" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    const n_cols: usize = 4;
    const bs: usize = 32;

    var cache = try QuantizedKVCache(f32).init(alloc, d_head, n_cols, bs);
    defer cache.deinit(alloc);

    var src: [d_head]f32 = undefined;
    fillRandF32(&src, 1);
    cache.storeColumn(2, &src);

    var out: [d_head]f32 = undefined;
    cache.dequantColumn(2, &out);

    for (src, out) |s, o| try testing.expectApproxEqAbs(s, o, 0.02);
}

test "QuantizedKVCache - dotF32 matches reference" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    var cache = try QuantizedKVCache(f32).init(alloc, d_head, 1, 32);
    defer cache.deinit(alloc);

    var k_col: [d_head]f32 = undefined;
    var q_vec: [d_head]f32 = undefined;
    fillRandF32(&k_col, 2);
    fillRandF32(&q_vec, 3);
    cache.storeColumn(0, &k_col);

    var ref: f32 = 0;
    for (q_vec, k_col) |a, b| ref += a * b;

    const got = cache.dotF32(&q_vec, 0);
    try testing.expectApproxEqAbs(ref, got, 0.05);
}

test "QuantizedKVCache - dotI8 (SDOT path) matches dotF32" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    const bs: usize = 32;
    var cache = try QuantizedKVCache(f32).init(alloc, d_head, 1, bs);
    defer cache.deinit(alloc);

    var k_col: [d_head]f32 = undefined;
    var q_vec: [d_head]f32 = undefined;
    fillRandF32(&k_col, 42);
    fillRandF32(&q_vec, 43);
    cache.storeColumn(0, &k_col);

    var q_i8: [d_head]i8 = undefined;
    var q_scales: [d_head / bs]f32 = undefined;
    QuantizedWeight(f32).quantizeInput(&q_vec, d_head, bs, &q_i8, &q_scales);

    const ref = cache.dotF32(&q_vec, 0);
    const got = cache.dotI8(&q_i8, &q_scales, 0);
    try testing.expectApproxEqAbs(ref, got, 0.05);
}

test "QuantizedKVCache - accumF32 matches reference" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    var cache = try QuantizedKVCache(f32).init(alloc, d_head, 1, 32);
    defer cache.deinit(alloc);

    var v_col: [d_head]f32 = undefined;
    fillRandF32(&v_col, 4);
    cache.storeColumn(0, &v_col);

    const w: f32 = 0.7;
    var ref: [d_head]f32 = .{0} ** d_head;
    for (&ref, v_col) |*r, v| r.* = w * v;

    var got: [d_head]f32 = .{0} ** d_head;
    cache.accumF32(&got, w, 0);

    for (ref, got) |r, g| try testing.expectApproxEqAbs(r, g, 0.02);
}

test "attentionQuantized - decode (seq_q=1) matches reference" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    const seq_kv: usize = 8;
    const bs: usize = 32;

    var k_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer k_cache.deinit(alloc);
    var v_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer v_cache.deinit(alloc);

    var k_data: [d_head * seq_kv]f32 = undefined;
    var v_data: [d_head * seq_kv]f32 = undefined;
    fillRandF32(&k_data, 10);
    fillRandF32(&v_data, 11);

    for (0..seq_kv) |ci| {
        k_cache.storeColumn(ci, k_data[ci * d_head ..][0..d_head]);
        v_cache.storeColumn(ci, v_data[ci * d_head ..][0..d_head]);
    }

    var q: [d_head]f32 = undefined;
    fillRandF32(&q, 12);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    // Reference: dequantize K/V, run the same streaming softmax.
    var k_ref: [d_head * seq_kv]f32 = undefined;
    var v_ref: [d_head * seq_kv]f32 = undefined;
    for (0..seq_kv) |ci| {
        k_cache.dequantColumn(ci, k_ref[ci * d_head ..][0..d_head]);
        v_cache.dequantColumn(ci, v_ref[ci * d_head ..][0..d_head]);
    }
    var ref_out: [d_head]f32 = .{0} ** d_head;
    var m_val: f32 = -std.math.inf(f32);
    var l: f32 = 0;
    for (0..seq_kv) |s| {
        var dot: f32 = 0;
        for (0..d_head) |r| dot += q[r] * k_ref[s * d_head + r];
        const score = dot * scale;
        const new_m = @max(m_val, score);
        const alpha: f32 = if (m_val == -std.math.inf(f32)) 0 else @exp(m_val - new_m);
        const w = @exp(score - new_m);
        for (0..d_head) |r| ref_out[r] = ref_out[r] * alpha + w * v_ref[s * d_head + r];
        l = l * alpha + w;
        m_val = new_m;
    }
    const inv_l = if (l > 0) 1.0 / l else 0.0;
    for (&ref_out) |*r| r.* *= inv_l;

    var out: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32,
        &out,
        d_head,
        &q,
        d_head,
        d_head,
        1,
        &k_cache,
        0,
        &v_cache,
        0,
        seq_kv,
        null,
        0,
        0,
        scale,
    );

    // Tolerance accounts for extra Q-quantization error on the SDOT fast path.
    for (ref_out, out) |r, o| try testing.expectApproxEqAbs(r, o, 0.01);
}

test "attentionQuantized - causal mask via broadcast column" {
    const alloc = testing.allocator;
    const d_head: usize = 32;
    const seq_kv: usize = 8;
    const pos: usize = 4;
    const bs: usize = 32;

    var k_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer k_cache.deinit(alloc);
    var v_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer v_cache.deinit(alloc);

    var k_data: [d_head * seq_kv]f32 = undefined;
    var v_data: [d_head * seq_kv]f32 = undefined;
    fillRandF32(&k_data, 20);
    fillRandF32(&v_data, 21);
    for (0..seq_kv) |ci| {
        k_cache.storeColumn(ci, k_data[ci * d_head ..][0..d_head]);
        v_cache.storeColumn(ci, v_data[ci * d_head ..][0..d_head]);
    }

    var q: [d_head]f32 = undefined;
    fillRandF32(&q, 22);

    // Mask: 0 for positions <= pos, -inf elsewhere. Broadcast single column.
    var mask: [seq_kv]f32 = undefined;
    for (0..seq_kv) |i| mask[i] = if (i <= pos) 0 else -std.math.inf(f32);

    const scale: f32 = 0.25;

    var out_masked: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32,
        &out_masked,
        d_head,
        &q,
        d_head,
        d_head,
        1,
        &k_cache,
        0,
        &v_cache,
        0,
        seq_kv,
        &mask,
        1,
        0,
        scale,
    );

    // Expected: same result as attending only to positions [0..pos+1].
    var out_truncated: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32,
        &out_truncated,
        d_head,
        &q,
        d_head,
        d_head,
        1,
        &k_cache,
        0,
        &v_cache,
        0,
        pos + 1,
        null,
        0,
        0,
        scale,
    );

    for (out_masked, out_truncated) |m, t| try testing.expectApproxEqAbs(m, t, 1e-6);
}

test "attentionQuantized - col_offset selects correct slab" {
    const alloc = testing.allocator;
    const d_head: usize = 32;
    const slab_len: usize = 6;
    const n_slabs: usize = 2;
    const bs: usize = 32;

    // One consolidated cache with two slabs; write different random data into each.
    var k_big = try QuantizedKVCache(f32).init(alloc, d_head, slab_len * n_slabs, bs);
    defer k_big.deinit(alloc);
    var v_big = try QuantizedKVCache(f32).init(alloc, d_head, slab_len * n_slabs, bs);
    defer v_big.deinit(alloc);

    // Independent caches holding just slab 1.
    var k_small = try QuantizedKVCache(f32).init(alloc, d_head, slab_len, bs);
    defer k_small.deinit(alloc);
    var v_small = try QuantizedKVCache(f32).init(alloc, d_head, slab_len, bs);
    defer v_small.deinit(alloc);

    var k_s0: [d_head * slab_len]f32 = undefined;
    var v_s0: [d_head * slab_len]f32 = undefined;
    var k_s1: [d_head * slab_len]f32 = undefined;
    var v_s1: [d_head * slab_len]f32 = undefined;
    fillRandF32(&k_s0, 30);
    fillRandF32(&v_s0, 31);
    fillRandF32(&k_s1, 32);
    fillRandF32(&v_s1, 33);

    for (0..slab_len) |ci| {
        k_big.storeColumn(ci, k_s0[ci * d_head ..][0..d_head]);
        v_big.storeColumn(ci, v_s0[ci * d_head ..][0..d_head]);
        k_big.storeColumn(slab_len + ci, k_s1[ci * d_head ..][0..d_head]);
        v_big.storeColumn(slab_len + ci, v_s1[ci * d_head ..][0..d_head]);
        k_small.storeColumn(ci, k_s1[ci * d_head ..][0..d_head]);
        v_small.storeColumn(ci, v_s1[ci * d_head ..][0..d_head]);
    }

    var q: [d_head]f32 = undefined;
    fillRandF32(&q, 34);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    var out_big: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32, &out_big, d_head, &q, d_head,
        d_head, 1,
        &k_big, slab_len, &v_big, slab_len,
        slab_len, null, 0, 0, scale,
    );

    var out_small: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32, &out_small, d_head, &q, d_head,
        d_head, 1,
        &k_small, 0, &v_small, 0,
        slab_len, null, 0, 0, scale,
    );

    for (out_big, out_small) |b, s| try testing.expectApproxEqAbs(b, s, 1e-6);
}

test "attentionQuantized - tile + tail path matches single-column reference" {
    const alloc = testing.allocator;
    const d_head: usize = 64;
    // Several full Bs-tiles plus a non-zero tail exercises both code paths.
    const seq_kv: usize = 21;
    const bs: usize = 32;

    var k_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer k_cache.deinit(alloc);
    var v_cache = try QuantizedKVCache(f32).init(alloc, d_head, seq_kv, bs);
    defer v_cache.deinit(alloc);

    var k_data: [d_head * seq_kv]f32 = undefined;
    var v_data: [d_head * seq_kv]f32 = undefined;
    fillRandF32(&k_data, 40);
    fillRandF32(&v_data, 41);
    for (0..seq_kv) |ci| {
        k_cache.storeColumn(ci, k_data[ci * d_head ..][0..d_head]);
        v_cache.storeColumn(ci, v_data[ci * d_head ..][0..d_head]);
    }

    var q: [d_head]f32 = undefined;
    fillRandF32(&q, 42);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    // Reference: dequantize then streaming-softmax single-column.
    var k_ref: [d_head * seq_kv]f32 = undefined;
    var v_ref: [d_head * seq_kv]f32 = undefined;
    for (0..seq_kv) |ci| {
        k_cache.dequantColumn(ci, k_ref[ci * d_head ..][0..d_head]);
        v_cache.dequantColumn(ci, v_ref[ci * d_head ..][0..d_head]);
    }
    var ref_out: [d_head]f32 = .{0} ** d_head;
    var m_val: f32 = -std.math.inf(f32);
    var l: f32 = 0;
    for (0..seq_kv) |s| {
        var dot: f32 = 0;
        for (0..d_head) |r| dot += q[r] * k_ref[s * d_head + r];
        const score = dot * scale;
        const new_m = @max(m_val, score);
        const alpha: f32 = if (m_val == -std.math.inf(f32)) 0 else @exp(m_val - new_m);
        const w = @exp(score - new_m);
        for (0..d_head) |r| ref_out[r] = ref_out[r] * alpha + w * v_ref[s * d_head + r];
        l = l * alpha + w;
        m_val = new_m;
    }
    const inv_l = if (l > 0) 1.0 / l else 0.0;
    for (&ref_out) |*r| r.* *= inv_l;

    var out: [d_head]f32 = .{0} ** d_head;
    attentionQuantized(
        f32, &out, d_head, &q, d_head,
        d_head, 1,
        &k_cache, 0, &v_cache, 0,
        seq_kv, null, 0, 0, scale,
    );

    // Tolerance matches the SDOT-path test — Q-quantization adds error.
    for (ref_out, out) |r, o| try testing.expectApproxEqAbs(r, o, 0.01);
}
