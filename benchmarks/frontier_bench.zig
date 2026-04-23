//! Decision-grade benchmark frontier for zgml.
//!
//! Run with: zig build bench-frontier
//!
//! The harness adaptively batches each sample until the elapsed time is large
//! enough to avoid timer-resolution artifacts, then reports per-iteration
//! latency. Every sample checksums the output so optimized builds must keep the
//! computed values live.

const std = @import("std");
const opts = @import("zgml_options");

const Tensor = @import("zgml_tensor").Tensor;

const SampleCount = 15;
const WarmupSamples = 3;
const MinSampleNs: u64 = 2_000_000;
const MaxRepeats: usize = 1 << 20;

fn nowNs(io: std.Io) u64 {
    return @intCast(std.Io.Clock.awake.now(io).nanoseconds);
}

fn fillDeterministic(data: []f32, seed: u64, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (data) |*v| {
        v.* = (rng.float(f32) * 2.0 - 1.0) * scale;
    }
}

fn checksum(data: []const f32) f64 {
    var acc: f64 = 0;
    var finite_count: usize = 0;
    const stride = @max(@as(usize, 1), data.len / 4096);
    var i: usize = 0;
    while (i < data.len) : (i += stride) {
        const v = data[i];
        if (std.math.isFinite(v)) finite_count += 1;
        acc += @as(f64, @floatCast(v)) * @as(f64, @floatFromInt((i % 251) + 1));
    }
    if (finite_count == 0 or !std.math.isFinite(acc)) @panic("benchmark produced non-finite output");
    std.mem.doNotOptimizeAway(acc);
    return acc;
}

const BenchStats = struct {
    repeats: usize,
    min_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    checksum: f64,
};

fn calibrateRepeats(io: std.Io, bench: anytype) usize {
    var repeats: usize = 1;
    while (true) {
        const t0 = nowNs(io);
        for (0..repeats) |_| bench.run();
        const elapsed = nowNs(io) - t0;
        _ = bench.consume();
        if (elapsed >= MinSampleNs or repeats >= MaxRepeats) return repeats;
        repeats = @min(repeats * 2, MaxRepeats);
    }
}

fn measure(io: std.Io, bench: anytype) BenchStats {
    const repeats = calibrateRepeats(io, bench);
    for (0..WarmupSamples) |_| {
        for (0..repeats) |_| bench.run();
        _ = bench.consume();
    }

    var times: [SampleCount]u64 = undefined;
    var total_check: f64 = 0;
    for (&times) |*t| {
        const t0 = nowNs(io);
        for (0..repeats) |_| bench.run();
        t.* = nowNs(io) - t0;
        total_check += bench.consume();
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));

    const denom = @as(f64, @floatFromInt(repeats));
    return .{
        .repeats = repeats,
        .min_ns = @as(f64, @floatFromInt(times[0])) / denom,
        .p50_ns = @as(f64, @floatFromInt(times[SampleCount / 2])) / denom,
        .p90_ns = @as(f64, @floatFromInt(times[(SampleCount * 9) / 10])) / denom,
        .checksum = total_check,
    };
}

fn printStats(
    w: *std.Io.Writer,
    name: []const u8,
    work_label: []const u8,
    work_units: f64,
    unit_suffix: []const u8,
    stats: BenchStats,
) !void {
    const seconds = stats.p50_ns / 1_000_000_000.0;
    const throughput = if (seconds > 0) work_units / seconds else 0;
    try w.print(
        "  {s:<28} p50={d:>10.1} ns  min={d:>10.1}  p90={d:>10.1}  reps={d:<7}  {s}={d:>9.2} {s}/s  check={d:.3}\n",
        .{ name, stats.p50_ns, stats.min_ns, stats.p90_ns, stats.repeats, work_label, throughput, unit_suffix, stats.checksum },
    );
}

const TensorComputeBench = struct {
    out: *Tensor(f32),

    fn run(self: *TensorComputeBench) void {
        self.out.compute();
    }

    fn consume(self: *TensorComputeBench) f64 {
        return checksum(self.out.data);
    }
};

const MatMulBench = struct {
    out: *Tensor(f32),

    fn run(self: *MatMulBench) void {
        self.out.compute();
    }

    fn consume(self: *MatMulBench) f64 {
        return checksum(self.out.data);
    }
};

const ChainBench = struct {
    x: *Tensor(f32),
    y: *Tensor(f32),
    bias: *Tensor(f32),
    tmp0: *Tensor(f32),
    tmp1: *Tensor(f32),
    tmp2: *Tensor(f32),
    tmp3: *Tensor(f32),
    out: *Tensor(f32),

    fn init(alloc: std.mem.Allocator, n: usize) !ChainBench {
        const x = try Tensor(f32).init(alloc, &.{n});
        errdefer x.deinit();
        const y = try Tensor(f32).init(alloc, &.{n});
        errdefer y.deinit();
        const bias = try Tensor(f32).init(alloc, &.{n});
        errdefer bias.deinit();
        const tmp0 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp0.deinit();
        const tmp1 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp1.deinit();
        const tmp2 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp2.deinit();
        const tmp3 = try Tensor(f32).init(alloc, &.{n});
        errdefer tmp3.deinit();
        const out = try Tensor(f32).init(alloc, &.{n});
        errdefer out.deinit();

        fillDeterministic(x.data, 101, 0.6);
        fillDeterministic(y.data, 102, 0.4);
        fillDeterministic(bias.data, 103, 0.2);

        return .{ .x = x, .y = y, .bias = bias, .tmp0 = tmp0, .tmp1 = tmp1, .tmp2 = tmp2, .tmp3 = tmp3, .out = out };
    }

    fn deinit(self: *ChainBench) void {
        self.x.deinit();
        self.y.deinit();
        self.bias.deinit();
        self.tmp0.deinit();
        self.tmp1.deinit();
        self.tmp2.deinit();
        self.tmp3.deinit();
        self.out.deinit();
    }

    fn run(self: *ChainBench) void {
        self.tmp0.computeMul(self.x, self.y);
        self.tmp1.computeAdd(self.tmp0, self.bias);
        self.tmp2.computeRelu(self.tmp1);
        self.tmp3.computeMul(self.tmp2, self.y);
        self.out.computeAdd(self.tmp3, self.x);
    }

    fn consume(self: *ChainBench) f64 {
        return checksum(self.out.data);
    }
};

const FusedChainBench = struct {
    x: *Tensor(f32),
    y: *Tensor(f32),
    bias: *Tensor(f32),
    out: *Tensor(f32),

    fn init(alloc: std.mem.Allocator, n: usize) !FusedChainBench {
        const x = try Tensor(f32).init(alloc, &.{n});
        errdefer x.deinit();
        const y = try Tensor(f32).init(alloc, &.{n});
        errdefer y.deinit();
        const bias = try Tensor(f32).init(alloc, &.{n});
        errdefer bias.deinit();
        const out = try Tensor(f32).init(alloc, &.{n});
        errdefer out.deinit();

        fillDeterministic(x.data, 101, 0.6);
        fillDeterministic(y.data, 102, 0.4);
        fillDeterministic(bias.data, 103, 0.2);

        return .{ .x = x, .y = y, .bias = bias, .out = out };
    }

    fn deinit(self: *FusedChainBench) void {
        self.x.deinit();
        self.y.deinit();
        self.bias.deinit();
        self.out.deinit();
    }

    fn run(self: *FusedChainBench) void {
        const Vec = @Vector(8, f32);
        const zero: Vec = @splat(0);
        var i: usize = 0;
        while (i + 8 <= self.out.data.len) : (i += 8) {
            const xv: Vec = self.x.data[i..][0..8].*;
            const yv: Vec = self.y.data[i..][0..8].*;
            const bv: Vec = self.bias.data[i..][0..8].*;
            self.out.data[i..][0..8].* = @max(xv * yv + bv, zero) * yv + xv;
        }
        while (i < self.out.data.len) : (i += 1) {
            const x = self.x.data[i];
            const y = self.y.data[i];
            const h = @max(x * y + self.bias.data[i], 0);
            self.out.data[i] = h * y + x;
        }
    }

    fn consume(self: *FusedChainBench) f64 {
        return checksum(self.out.data);
    }
};

fn benchElementwise(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer) !void {
    try w.print("\nElementwise Chain And Fusion\n", .{});
    try w.print("----------------------------\n", .{});

    const sizes = [_]usize{ 4_096, 262_144 };
    for (sizes) |n| {
        var unfused = try ChainBench.init(alloc, n);
        defer unfused.deinit();
        var fused = try FusedChainBench.init(alloc, n);
        defer fused.deinit();

        const unfused_stats = measure(io, &unfused);
        const fused_stats = measure(io, &fused);
        const elems = @as(f64, @floatFromInt(n));
        var name_buf: [64]u8 = undefined;

        const unfused_name = try std.fmt.bufPrint(&name_buf, "chain n={d} staged", .{n});
        try printStats(w, unfused_name, "elems", elems, "elem", unfused_stats);

        const fused_name = try std.fmt.bufPrint(&name_buf, "chain n={d} one-pass", .{n});
        try printStats(w, fused_name, "elems", elems, "elem", fused_stats);
    }
}

const MatmulCase = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    trans_b: bool = false,
};

fn benchMatmul(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer) !void {
    try w.print("\nMatmul Shape Regimes\n", .{});
    try w.print("--------------------\n", .{});

    const cases = [_]MatmulCase{
        .{ .name = "decode gemv", .m = 1, .n = 512, .k = 256 },
        .{ .name = "small square", .m = 128, .n = 128, .k = 128 },
        .{ .name = "batched projection", .m = 32, .n = 512, .k = 256 },
        .{ .name = "attention scores", .m = 64, .n = 128, .k = 64, .trans_b = true },
    };

    for (cases) |case| {
        const a = try Tensor(f32).init(alloc, &.{ case.k, case.m });
        defer a.deinit();
        const b_shape = if (case.trans_b) [_]usize{ case.k, case.n } else [_]usize{ case.n, case.k };
        const b = try Tensor(f32).init(alloc, &b_shape);
        defer b.deinit();
        fillDeterministic(a.data, 201, 0.08);
        fillDeterministic(b.data, 202, 0.08);

        const out = a.matMul(false, b, case.trans_b);
        defer out.deinit();
        var bench = MatMulBench{ .out = out };
        const stats = measure(io, &bench);
        const flops = 2.0 * @as(f64, @floatFromInt(case.m)) * @as(f64, @floatFromInt(case.n)) * @as(f64, @floatFromInt(case.k));
        try printStats(w, case.name, "throughput", flops / 1_000_000_000.0, "GFLOP", stats);
    }
}

fn benchNorms(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer) !void {
    try w.print("\nSoftmax And RMSNorm\n", .{});
    try w.print("-------------------\n", .{});

    {
        const rows: usize = 1024;
        const cols: usize = 32;
        const logits = try Tensor(f32).init(alloc, &.{ rows, cols });
        defer logits.deinit();
        fillDeterministic(logits.data, 301, 2.0);
        const out = logits.softmax(&.{ 1, cols });
        defer out.deinit();
        var bench = TensorComputeBench{ .out = out };
        const stats = measure(io, &bench);
        try printStats(w, "softmax 1024 x 32", "elems", @floatFromInt(rows * cols), "elem", stats);
    }

    {
        const hidden: usize = 768;
        const tokens: usize = 64;
        const x = try Tensor(f32).init(alloc, &.{ hidden, tokens });
        defer x.deinit();
        fillDeterministic(x.data, 302, 0.5);
        const out = x.rmsNorm(&.{ 1, tokens }, 1e-5);
        defer out.deinit();
        var bench = TensorComputeBench{ .out = out };
        const stats = measure(io, &bench);
        try printStats(w, "rmsnorm 768 x 64", "elems", @floatFromInt(hidden * tokens), "elem", stats);
    }
}

const DecodeBench = struct {
    x: *Tensor(f32),
    norm_w: *Tensor(f32),
    wq: *Tensor(f32),
    k_cache: *Tensor(f32),
    v_cache: *Tensor(f32),
    mask: *Tensor(f32),
    wout: *Tensor(f32),
    rms: *Tensor(f32),
    norm: *Tensor(f32),
    q: *Tensor(f32),
    scores: *Tensor(f32),
    masked: *Tensor(f32),
    probs: *Tensor(f32),
    context: *Tensor(f32),
    logits: *Tensor(f32),

    fn init(alloc: std.mem.Allocator) !DecodeBench {
        const d_model: usize = 64;
        const seq: usize = 32;
        const vocab: usize = 512;

        const x = try Tensor(f32).init(alloc, &.{ d_model, 1 });
        errdefer x.deinit();
        const norm_w = try Tensor(f32).init(alloc, &.{ d_model, 1 });
        errdefer norm_w.deinit();
        const wq = try Tensor(f32).init(alloc, &.{ d_model, d_model });
        errdefer wq.deinit();
        const k_cache = try Tensor(f32).init(alloc, &.{ d_model, seq });
        errdefer k_cache.deinit();
        const v_cache = try Tensor(f32).init(alloc, &.{ d_model, seq });
        errdefer v_cache.deinit();
        const mask = try Tensor(f32).init(alloc, &.{ seq, 1 });
        errdefer mask.deinit();
        const wout = try Tensor(f32).init(alloc, &.{ vocab, d_model });
        errdefer wout.deinit();

        fillDeterministic(x.data, 401, 0.25);
        _ = norm_w.setAllScalar(1);
        fillDeterministic(wq.data, 402, 0.08);
        fillDeterministic(k_cache.data, 403, 0.05);
        fillDeterministic(v_cache.data, 404, 0.05);
        for (mask.data, 0..) |*v, i| v.* = if (i <= 15) 0 else -1e9;
        fillDeterministic(wout.data, 405, 0.08);

        const rms = x.rmsNorm(&.{ 1, 1 }, 1e-5);
        errdefer rms.deinit();
        const norm = rms.mul(norm_w);
        errdefer norm.deinit();
        const q = norm.matMul(false, wq, false);
        errdefer q.deinit();
        const scores0 = q.matMul(false, k_cache, true);
        errdefer scores0.deinit();
        const masked = scores0.add(mask);
        errdefer masked.deinit();
        const probs = masked.softmax(&.{ 1, 1 });
        errdefer probs.deinit();
        const context = probs.matMul(false, v_cache, false);
        errdefer context.deinit();
        const logits = context.matMul(false, wout, false);
        errdefer logits.deinit();

        return .{
            .x = x,
            .norm_w = norm_w,
            .wq = wq,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .mask = mask,
            .wout = wout,
            .rms = rms,
            .norm = norm,
            .q = q,
            .scores = scores0,
            .masked = masked,
            .probs = probs,
            .context = context,
            .logits = logits,
        };
    }

    fn deinit(self: *DecodeBench) void {
        self.logits.deinit();
        self.context.deinit();
        self.probs.deinit();
        self.masked.deinit();
        self.scores.deinit();
        self.q.deinit();
        self.norm.deinit();
        self.rms.deinit();
        self.wout.deinit();
        self.mask.deinit();
        self.v_cache.deinit();
        self.k_cache.deinit();
        self.wq.deinit();
        self.norm_w.deinit();
        self.x.deinit();
    }

    fn run(self: *DecodeBench) void {
        self.rms.compute();
        self.norm.compute();
        self.q.compute();
        self.scores.compute();
        self.masked.compute();
        self.probs.compute();
        self.context.compute();
        self.logits.compute();
    }

    fn consume(self: *DecodeBench) f64 {
        return checksum(self.logits.data);
    }
};

fn benchDecodeGraph(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer) !void {
    try w.print("\nDecode-ish Inference Path\n", .{});
    try w.print("-------------------------\n", .{});

    var bench = try DecodeBench.init(alloc);
    defer bench.deinit();
    const stats = measure(io, &bench);
    try printStats(w, "rmsnorm-attn-logits token", "tokens", 1.0, "tok", stats);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const alloc = init.gpa;

    const stdout_file = std.Io.File.stdout();
    var buf: [16 * 1024]u8 = undefined;
    var writer = stdout_file.writer(io, &buf);
    const w = &writer.interface;

    try w.print("\nzgml benchmark frontier", .{});
    if (opts.use_blas) try w.print(" [BLAS enabled]", .{});
    try w.print("\n=======================\n", .{});
    try w.print("samples={d}, min_sample={d:.1} ms, adaptive repeats, ReleaseFast build step\n", .{
        SampleCount,
        @as(f64, @floatFromInt(MinSampleNs)) / 1_000_000.0,
    });

    try benchElementwise(io, alloc, w);
    try benchMatmul(io, alloc, w);
    try benchNorms(io, alloc, w);
    try benchDecodeGraph(io, alloc, w);

    try w.print("\n", .{});
    writer.interface.flush() catch {};
}
