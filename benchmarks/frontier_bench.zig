//! Decision-grade benchmark frontier for internal.
//!
//! Run with: zig build bench-frontier
//!
//! The harness adaptively batches each sample until the elapsed time is large
//! enough to avoid timer-resolution artifacts, then reports per-iteration
//! latency. Every sample checksums the output so optimized builds must keep the
//! computed values live.

const std = @import("std");
const opts = @import("zgml_options");
const internal = @import("zgml_internal");
const backend_mod = internal.backend;
const program_mod = internal.backend_program;
const profile_mod = internal.profile;

const Tensor = internal.Tensor;

const SampleCount = 15;
const WarmupSamples = 3;
const MinSampleNs: u64 = 8_000_000;
const MinRepeats: usize = 8;
const MaxRepeats: usize = 1 << 20;
const FrontierStencilVecLen = 16;

const FrontierFilter = struct {
    query: ?[]const u8,

    fn init() FrontierFilter {
        const raw = std.c.getenv("BENCH_FRONTIER_FILTER") orelse return .{ .query = null };
        const query = std.mem.span(raw);
        return .{ .query = if (query.len == 0) null else query };
    }

    fn enabled(self: FrontierFilter) bool {
        return self.query != null;
    }

    fn matches(self: FrontierFilter, name: []const u8) bool {
        const query = self.query orelse return true;
        return std.mem.indexOf(u8, name, query) != null;
    }

    fn matchesAny(self: FrontierFilter, names: []const []const u8) bool {
        if (!self.enabled()) return true;
        for (names) |name| {
            if (self.matches(name)) return true;
        }
        return false;
    }
};

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

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) @panic("mismatched benchmark outputs");
    var max_diff: f32 = 0;
    for (a, b) |x, y| {
        const diff = @abs(x - y);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

const BenchStats = struct {
    repeats: usize,
    min_ns: f64,
    p50_ns: f64,
    p90_ns: f64,
    checksum: f64,
};

fn calibrateRepeats(io: std.Io, bench: anytype) usize {
    var repeats: usize = MinRepeats;
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

fn printRatio(
    w: *std.Io.Writer,
    name: []const u8,
    numerator: BenchStats,
    denominator: BenchStats,
    diff: ?f32,
) !void {
    const ratio = numerator.p50_ns / denominator.p50_ns;
    if (diff) |max_diff| {
        try w.print("  {s:<28} speedup={d:.2}x  old_p50={d:.1} ns  fused_p50={d:.1} ns  max_abs_diff={d:.6}\n", .{
            name,
            ratio,
            numerator.p50_ns,
            denominator.p50_ns,
            max_diff,
        });
    } else {
        try w.print("  {s:<28} speedup={d:.2}x  old_p50={d:.1} ns  fused_p50={d:.1} ns\n", .{
            name,
            ratio,
            numerator.p50_ns,
            denominator.p50_ns,
        });
    }
}

fn printCommandShape(
    w: *std.Io.Writer,
    name: []const u8,
    commands: []const program_mod.ProgramCommand,
) !void {
    const shape = try program_mod.ProgramCommandStreamShape.fromCommands(commands);
    try w.print(
        "  {s:<28} shape_commands={d}  shape_projection_row_chains={d}  shape_covered_ops={d}  shape_saved_dispatches={d}  shape_projection_groups={d}\n",
        .{
            name,
            shape.command_count,
            shape.projection_row_chains,
            shape.covered_ops,
            shape.estimated_saved_dispatches,
            shape.projection_groups,
        },
    );
}

fn printProjectionRowChainRuntimeProfile(
    w: *std.Io.Writer,
    name: []const u8,
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    output_io: []const backend_mod.ProgramIO,
) !void {
    be.resetRuntimeProfile(handle);
    be.executeProgram(handle, &.{}, output_io);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    try w.print(
        "  {s:<28} runtime_command_dispatches={d}  qmatmul_row_chain_tiled_two_phase_count={d}  qmatmul_row_chain_tiled_spilled_elementwise={d}\n",
        .{
            name,
            rt.backend_dispatch_count,
            rt.qmatmul_row_chain_tiled_two_phase_count,
            rt.qmatmul_row_chain_tiled_spilled_elementwise,
        },
    );
}

fn printProjectionGroupRuntimeProfile(
    w: *std.Io.Writer,
    name: []const u8,
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    output_io: []const backend_mod.ProgramIO,
) !void {
    be.resetRuntimeProfile(handle);
    be.executeProgram(handle, &.{}, output_io);
    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeProfileTo(handle, &rt);
    try w.print(
        "  {s:<28} runtime_backend_dispatches={d}  runtime_projection_group_dispatches={d}  runtime_projection_cache_group_dispatches={d}  runtime_projection_group_count={d}  runtime_projection_cache_group_count={d}\n",
        .{
            name,
            rt.backend_dispatch_count,
            rt.program_command_dispatch_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_group)],
            rt.program_command_dispatch_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_cache_group)],
            rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_group)],
            rt.program_command_counts[@intFromEnum(program_mod.ProgramCommandKind.projection_cache_group)],
        },
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
        const vec_len = FrontierStencilVecLen;
        const Vec = @Vector(vec_len, f32);
        const zero: Vec = @splat(0);
        var i: usize = 0;
        while (i + vec_len <= self.out.data.len) : (i += vec_len) {
            const xv: Vec = self.x.data[i..][0..vec_len].*;
            const yv: Vec = self.y.data[i..][0..vec_len].*;
            const bv: Vec = self.bias.data[i..][0..vec_len].*;
            self.out.data[i..][0..vec_len].* = @max(xv * yv + bv, zero) * yv + xv;
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

fn benchElementwise(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Elementwise", "chain" })) return;
    try w.print("\nElementwise Chain And Fusion\n", .{});
    try w.print("----------------------------\n", .{});

    const sizes = [_]usize{ 4_096, 262_144 };
    for (sizes) |n| {
        var unfused_name_buf: [64]u8 = undefined;
        const unfused_name = try std.fmt.bufPrint(&unfused_name_buf, "chain n={d} staged", .{n});
        var fused_name_buf: [64]u8 = undefined;
        const fused_name = try std.fmt.bufPrint(&fused_name_buf, "chain n={d} one-pass", .{n});
        if (!filter.matchesAny(&.{ unfused_name, fused_name })) continue;

        var unfused = try ChainBench.init(alloc, n);
        defer unfused.deinit();
        var fused = try FusedChainBench.init(alloc, n);
        defer fused.deinit();

        const unfused_stats = measure(io, &unfused);
        const fused_stats = measure(io, &fused);
        const elems = @as(f64, @floatFromInt(n));
        try printStats(w, unfused_name, "elems", elems, "elem", unfused_stats);

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

fn benchMatmul(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Matmul", "gemv", "projection", "attention" })) return;
    try w.print("\nMatmul Shape Regimes\n", .{});
    try w.print("--------------------\n", .{});

    const cases = [_]MatmulCase{
        .{ .name = "decode gemv", .m = 1, .n = 512, .k = 256 },
        .{ .name = "small square", .m = 128, .n = 128, .k = 128 },
        .{ .name = "batched projection", .m = 32, .n = 512, .k = 256 },
        .{ .name = "attention scores", .m = 64, .n = 128, .k = 64, .trans_b = true },
    };

    for (cases) |case| {
        if (!filter.matches(case.name)) continue;
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

fn benchNorms(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Softmax", "RMSNorm", "softmax", "rmsnorm" })) return;
    try w.print("\nSoftmax And RMSNorm\n", .{});
    try w.print("-------------------\n", .{});

    if (filter.matches("softmax 1024 x 32")) {
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

    if (filter.matches("rmsnorm 768 x 64")) {
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

fn benchDecodeGraph(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{ "Decode", "rmsnorm-attn-logits token" })) return;
    try w.print("\nDecode-ish Inference Path\n", .{});
    try w.print("-------------------------\n", .{});

    var bench = try DecodeBench.init(alloc);
    defer bench.deinit();
    const stats = measure(io, &bench);
    try printStats(w, "rmsnorm-attn-logits token", "tokens", 1.0, "tok", stats);
}

const ProjectionRowChainMetalBench = struct {
    be: backend_mod.Backend,
    handle: backend_mod.Backend.CompiledHandle,
    out: []f32,
    output_io: []const backend_mod.ProgramIO,

    fn run(self: *ProjectionRowChainMetalBench) void {
        self.be.executeProgram(self.handle, &.{}, self.output_io);
    }

    fn consume(self: *ProjectionRowChainMetalBench) f64 {
        return checksum(self.out);
    }
};

const ProjectionRowChainCase = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
};

fn allocF32(alloc: std.mem.Allocator, n: usize, seed: u64, scale: f32) ![]f32 {
    const data = try alloc.alloc(f32, n);
    fillDeterministic(data, seed, scale);
    return data;
}

fn allocI8Weights(alloc: std.mem.Allocator, n: usize, seed: u64) ![]i8 {
    const data = try alloc.alloc(i8, n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (data) |*v| {
        v.* = @intCast(rng.intRangeAtMost(i16, -12, 12));
    }
    return data;
}

fn programIo(buf_idx: u16, data: anytype) backend_mod.ProgramIO {
    return .{
        .buf_idx = buf_idx,
        .host_ptr = @ptrCast(data.ptr),
        .size = @intCast(data.len * @sizeOf(@TypeOf(data[0]))),
    };
}

fn benchProjectionChainMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 601, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, elems, 602, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, elems, 603, 0.20);
    defer alloc.free(residual);
    const ew_out = try allocF32(alloc, elems, 604, 0.0);
    defer alloc.free(ew_out);
    const default_out = try allocF32(alloc, elems, 605, 0.0);
    defer alloc.free(default_out);
    const fused_out = try allocF32(alloc, elems, 606, 0.0);
    defer alloc.free(fused_out);
    const qdata = try allocI8Weights(alloc, case.k * case.n, 607);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, scale_len, 608, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .add, .dst = 3, .src0 = 1, .src1 = 2, .n = @intCast(elems) } },
    };
    const buffer_sizes = [_]usize{ input_len, elems, elems, elems };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, q_out),
        programIo(2, residual),
        programIo(3, ew_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = qdata,
        .scales = qscales,
        .rows = case.k,
        .cols = case.n,
        .block_size = block_size,
    }};
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const fused_handle = metal.compileProgramWithCommandPolicy(program, program_mod.CommandStreamPolicy.default()) orelse return error.CompileFailed;
    defer be.freeProgram(fused_handle);

    const staged_output_io = [_]backend_mod.ProgramIO{programIo(3, default_out)};
    const fused_output_io = [_]backend_mod.ProgramIO{programIo(3, fused_out)};
    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = default_out,
        .output_io = &staged_output_io,
    };
    var fused_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = fused_handle,
        .out = fused_out,
        .output_io = &fused_output_io,
    };

    const staged_stats = measure(io, &staged_bench);
    const fused_stats = measure(io, &fused_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * case.n * case.k));

    var staged_name_buf: [96]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var fused_name_buf: [96]u8 = undefined;
    const fused_name = try std.fmt.bufPrint(&fused_name_buf, "{s} projection_chain", .{case.name});
    try printStats(w, fused_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", fused_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_chain", .{case.name});
    try printRatio(w, ratio_name, staged_stats, fused_stats, maxAbsDiff(default_out, fused_out));
}

fn benchProjectionGroupMetalCase(
    comptime n_slots: usize,
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 701, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, n_slots * elems, 702, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, n_slots * elems, 703, 0.20);
    defer alloc.free(residual);
    const staged_out = try allocF32(alloc, n_slots * elems, 704, 0.0);
    defer alloc.free(staged_out);
    const grouped_out = try allocF32(alloc, n_slots * elems, 705, 0.0);
    defer alloc.free(grouped_out);
    const qdata = try allocI8Weights(alloc, n_slots * case.k * case.n, 706);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, n_slots * scale_len, 707, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    var ops: [n_slots * 2]backend_mod.DeviceOp = undefined;
    var buffer_sizes: [1 + n_slots * 3]usize = undefined;
    var uploads: [1 + n_slots * 3]backend_mod.ProgramIO = undefined;
    var qweights: [n_slots]backend_mod.QuantizedWeightUpload = undefined;

    buffer_sizes[0] = input_len;
    uploads[0] = programIo(0, input);
    for (0..n_slots) |slot| {
        const q_buf: u16 = @intCast(1 + slot * 3);
        const residual_buf: u16 = @intCast(2 + slot * 3);
        const out_buf: u16 = @intCast(3 + slot * 3);
        const base = slot * elems;
        const weight_base = slot * case.k * case.n;
        const scale_base = slot * scale_len;

        ops[slot * 2] = .{ .qmatmul = .{ .dst = q_buf, .input = 0, .weight_idx = @intCast(slot), .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } };
        ops[slot * 2 + 1] = .{ .elementwise = .{ .op = .add, .dst = out_buf, .src0 = q_buf, .src1 = residual_buf, .n = @intCast(elems) } };

        buffer_sizes[1 + slot * 3] = elems;
        buffer_sizes[2 + slot * 3] = elems;
        buffer_sizes[3 + slot * 3] = elems;
        uploads[1 + slot * 3] = programIo(q_buf, q_out[base .. base + elems]);
        uploads[2 + slot * 3] = programIo(residual_buf, residual[base .. base + elems]);
        uploads[3 + slot * 3] = programIo(out_buf, staged_out[base .. base + elems]);
        qweights[slot] = .{
            .data = qdata[weight_base .. weight_base + case.k * case.n],
            .scales = qscales[scale_base .. scale_base + scale_len],
            .rows = case.k,
            .cols = case.n,
            .block_size = block_size,
        };
    }

    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.qmatmul_group_size = 1;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    const grouped_handle = metal.compileProgramWithCommandPolicy(program, program_mod.CommandStreamPolicy.default()) orelse return error.CompileFailed;
    defer be.freeProgram(grouped_handle);

    var staged_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var grouped_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    for (0..n_slots) |slot| {
        const out_buf: u16 = @intCast(3 + slot * 3);
        const base = slot * elems;
        staged_outputs[slot] = programIo(out_buf, staged_out[base .. base + elems]);
        grouped_outputs[slot] = programIo(out_buf, grouped_out[base .. base + elems]);
    }

    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = staged_out,
        .output_io = &staged_outputs,
    };
    var grouped_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = grouped_handle,
        .out = grouped_out,
        .output_io = &grouped_outputs,
    };

    const staged_stats = measure(io, &staged_bench);
    const grouped_stats = measure(io, &grouped_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(n_slots * case.m * case.n * case.k));

    var staged_name_buf: [96]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var grouped_name_buf: [96]u8 = undefined;
    const grouped_name = try std.fmt.bufPrint(&grouped_name_buf, "{s} projection_group", .{case.name});
    try printStats(w, grouped_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", grouped_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_group", .{case.name});
    try printRatio(w, ratio_name, staged_stats, grouped_stats, maxAbsDiff(staged_out, grouped_out));

    const grouped_commands = try program_mod.buildProgramCommands(alloc, &ops, program_mod.CommandStreamPolicy.default());
    defer alloc.free(grouped_commands);
    var profile_name_buf: [112]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_group dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, grouped_commands);
    try printProjectionGroupRuntimeProfile(w, profile_name, be, grouped_handle, &grouped_outputs);
}

fn benchProjectionRowChainMetalCase(
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 501, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, elems, 502, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, elems, 503, 0.20);
    defer alloc.free(residual);
    const norm_out = try allocF32(alloc, elems, 504, 0.0);
    defer alloc.free(norm_out);
    const scale = try allocF32(alloc, case.n, 505, 0.30);
    defer alloc.free(scale);
    const repeat_out = try allocF32(alloc, elems, 506, 0.0);
    defer alloc.free(repeat_out);
    const default_out = try allocF32(alloc, elems, 507, 0.0);
    defer alloc.free(default_out);
    const fused_out = try allocF32(alloc, elems, 508, 0.0);
    defer alloc.free(fused_out);
    const single_dispatch_out = try allocF32(alloc, elems, 511, 0.0);
    defer alloc.free(single_dispatch_out);
    const two_phase_out = try allocF32(alloc, elems, 512, 0.0);
    defer alloc.free(two_phase_out);
    const qdata = try allocI8Weights(alloc, case.k * case.n, 509);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, scale_len, 510, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{ .dst = 1, .input = 0, .weight_idx = 0, .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } },
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 1, .src1 = 2, .n = @intCast(elems) } },
        .{ .rmsnorm = .{ .dst = 3, .src = 2, .rows = @intCast(case.m), .cols = @intCast(case.n), .eps = 1e-5 } },
        .{ .repeat = .{
            .dst = 5,
            .src = 4,
            .n = @intCast(elems),
            .src_ne = .{ @intCast(case.n), 1, 1, 1 },
            .dst_ne = .{ @intCast(case.n), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(case.n), @intCast(case.n), @intCast(case.n) },
            .dst_strides = .{ 1, @intCast(case.n), @intCast(elems), @intCast(elems) },
        } },
        .{ .elementwise = .{ .op = .mul, .dst = 6, .src0 = 3, .src1 = 5, .n = @intCast(elems) } },
    };
    const buffer_sizes = [_]usize{ input_len, elems, elems, elems, case.n, elems, elems };
    const uploads = [_]backend_mod.ProgramIO{
        programIo(0, input),
        programIo(1, q_out),
        programIo(2, residual),
        programIo(3, norm_out),
        programIo(4, scale),
        programIo(5, repeat_out),
        programIo(6, default_out),
    };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = qdata,
        .scales = qscales,
        .rows = case.k,
        .cols = case.n,
        .block_size = block_size,
    }};
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var legacy_policy = program_mod.CommandStreamPolicy.default();
    legacy_policy.fuse_projection_row_chain = false;
    const default_handle = metal.compileProgramWithCommandPolicy(program, legacy_policy) orelse return error.CompileFailed;
    defer be.freeProgram(default_handle);
    var fused_policy = program_mod.CommandStreamPolicy.default();
    fused_policy.fuse_projection_row_chain = true;
    fused_policy.fuse_projection_row_chain_qmatvec = true;
    const fused_handle = metal.compileProgramWithCommandPolicy(program, fused_policy) orelse return error.CompileFailed;
    defer be.freeProgram(fused_handle);
    var single_dispatch_policy = fused_policy;
    single_dispatch_policy.fuse_projection_row_chain_single_dispatch = true;
    const single_dispatch_handle = metal.compileProgramWithCommandPolicy(program, single_dispatch_policy) orelse return error.CompileFailed;
    defer be.freeProgram(single_dispatch_handle);
    const two_phase_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainTwoPhaseCandidate();
    const two_phase_handle = metal.compileProgramWithCommandPolicy(program, two_phase_policy) orelse return error.CompileFailed;
    defer be.freeProgram(two_phase_handle);

    const default_output_io = [_]backend_mod.ProgramIO{programIo(6, default_out)};
    const fused_output_io = [_]backend_mod.ProgramIO{programIo(6, fused_out)};
    const single_dispatch_output_io = [_]backend_mod.ProgramIO{programIo(6, single_dispatch_out)};
    const two_phase_output_io = [_]backend_mod.ProgramIO{programIo(6, two_phase_out)};
    var default_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = default_handle,
        .out = default_out,
        .output_io = &default_output_io,
    };
    var fused_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = fused_handle,
        .out = fused_out,
        .output_io = &fused_output_io,
    };
    var single_dispatch_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = single_dispatch_handle,
        .out = single_dispatch_out,
        .output_io = &single_dispatch_output_io,
    };
    var two_phase_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = two_phase_handle,
        .out = two_phase_out,
        .output_io = &two_phase_output_io,
    };

    const default_stats = measure(io, &default_bench);
    const fused_stats = measure(io, &fused_bench);
    const single_dispatch_stats = measure(io, &single_dispatch_bench);
    const two_phase_stats = measure(io, &two_phase_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(case.m * case.n * case.k));

    var default_name_buf: [96]u8 = undefined;
    const default_name = try std.fmt.bufPrint(&default_name_buf, "{s} default", .{case.name});
    try printStats(w, default_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", default_stats);

    var fused_name_buf: [96]u8 = undefined;
    const fused_name = try std.fmt.bufPrint(&fused_name_buf, "{s} projection_row_chain", .{case.name});
    try printStats(w, fused_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", fused_stats);

    var single_dispatch_name_buf: [112]u8 = undefined;
    const single_dispatch_name = try std.fmt.bufPrint(&single_dispatch_name_buf, "{s} projection_row_chain_single_dispatch", .{case.name});
    try printStats(w, single_dispatch_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", single_dispatch_stats);

    var two_phase_name_buf: [112]u8 = undefined;
    const two_phase_name = try std.fmt.bufPrint(&two_phase_name_buf, "{s} projection_row_chain_two_phase", .{case.name});
    try printStats(w, two_phase_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", two_phase_stats);

    var ratio_name_buf: [96]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_row_chain", .{case.name});
    try printRatio(w, ratio_name, default_stats, fused_stats, maxAbsDiff(default_out, fused_out));

    var single_dispatch_ratio_name_buf: [128]u8 = undefined;
    const single_dispatch_ratio_name = try std.fmt.bufPrint(&single_dispatch_ratio_name_buf, "{s} projection_row_chain_single_dispatch", .{case.name});
    try printRatio(w, single_dispatch_ratio_name, default_stats, single_dispatch_stats, maxAbsDiff(default_out, single_dispatch_out));

    var two_phase_ratio_name_buf: [128]u8 = undefined;
    const two_phase_ratio_name = try std.fmt.bufPrint(&two_phase_ratio_name_buf, "{s} projection_row_chain_two_phase", .{case.name});
    try printRatio(w, two_phase_ratio_name, default_stats, two_phase_stats, maxAbsDiff(default_out, two_phase_out));

    const fused_commands = try program_mod.buildProgramCommands(alloc, &ops, fused_policy);
    defer alloc.free(fused_commands);
    var profile_name_buf: [112]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_row_chain dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, fused_commands);
    try printProjectionRowChainRuntimeProfile(w, profile_name, be, fused_handle, &fused_output_io);

    const single_dispatch_commands = try program_mod.buildProgramCommands(alloc, &ops, single_dispatch_policy);
    defer alloc.free(single_dispatch_commands);
    var single_dispatch_profile_name_buf: [128]u8 = undefined;
    const single_dispatch_profile_name = try std.fmt.bufPrint(&single_dispatch_profile_name_buf, "{s} projection_row_chain_single_dispatch dispatch_profile", .{case.name});
    try printCommandShape(w, single_dispatch_profile_name, single_dispatch_commands);
    try printProjectionRowChainRuntimeProfile(w, single_dispatch_profile_name, be, single_dispatch_handle, &single_dispatch_output_io);

    const two_phase_commands = try program_mod.buildProgramCommands(alloc, &ops, two_phase_policy);
    defer alloc.free(two_phase_commands);
    var two_phase_profile_name_buf: [128]u8 = undefined;
    const two_phase_profile_name = try std.fmt.bufPrint(&two_phase_profile_name_buf, "{s} projection_row_chain_two_phase dispatch_profile", .{case.name});
    try printCommandShape(w, two_phase_profile_name, two_phase_commands);
    try printProjectionRowChainRuntimeProfile(w, two_phase_profile_name, be, two_phase_handle, &two_phase_output_io);
}

fn benchProjectionRowChainGroupMetalCase(
    comptime n_slots: usize,
    io: std.Io,
    alloc: std.mem.Allocator,
    w: *std.Io.Writer,
    metal: *internal.backend_metal.MetalBackend,
    case: ProjectionRowChainCase,
) !void {
    const elems = case.m * case.n;
    const input_len = case.m * case.k;
    const block_size: usize = 32;
    const scale_len = (case.k * case.n + block_size - 1) / block_size;

    const input = try allocF32(alloc, input_len, 801, 0.25);
    defer alloc.free(input);
    const q_out = try allocF32(alloc, n_slots * elems, 802, 0.0);
    defer alloc.free(q_out);
    const residual = try allocF32(alloc, n_slots * elems, 803, 0.20);
    defer alloc.free(residual);
    const norm_out = try allocF32(alloc, n_slots * elems, 804, 0.0);
    defer alloc.free(norm_out);
    const scale = try allocF32(alloc, n_slots * case.n, 805, 0.30);
    defer alloc.free(scale);
    const repeat_out = try allocF32(alloc, n_slots * elems, 806, 0.0);
    defer alloc.free(repeat_out);
    const staged_out = try allocF32(alloc, n_slots * elems, 807, 0.0);
    defer alloc.free(staged_out);
    const grouped_out = try allocF32(alloc, n_slots * elems, 808, 0.0);
    defer alloc.free(grouped_out);
    const two_phase_out = try allocF32(alloc, n_slots * elems, 811, 0.0);
    defer alloc.free(two_phase_out);
    const qdata = try allocI8Weights(alloc, n_slots * case.k * case.n, 809);
    defer alloc.free(qdata);
    const qscales = try allocF32(alloc, n_slots * scale_len, 810, 0.02);
    defer alloc.free(qscales);
    for (qscales) |*v| v.* = 0.02;

    var ops: [n_slots * 5]backend_mod.DeviceOp = undefined;
    var buffer_sizes: [1 + n_slots * 6]usize = undefined;
    var uploads: [1 + n_slots * 6]backend_mod.ProgramIO = undefined;
    var qweights: [n_slots]backend_mod.QuantizedWeightUpload = undefined;

    buffer_sizes[0] = input_len;
    uploads[0] = programIo(0, input);
    for (0..n_slots) |slot| {
        const q_buf: u16 = @intCast(1 + slot * 6);
        const residual_buf: u16 = @intCast(2 + slot * 6);
        const norm_buf: u16 = @intCast(3 + slot * 6);
        const scale_buf: u16 = @intCast(4 + slot * 6);
        const repeat_buf: u16 = @intCast(5 + slot * 6);
        const out_buf: u16 = @intCast(6 + slot * 6);
        const elem_base = slot * elems;
        const scale_base = slot * case.n;
        const weight_base = slot * case.k * case.n;
        const qscale_base = slot * scale_len;
        const op_base = slot * 5;

        ops[op_base] = .{ .qmatmul = .{ .dst = q_buf, .input = 0, .weight_idx = @intCast(slot), .M = @intCast(case.m), .N = @intCast(case.n), .K = @intCast(case.k) } };
        ops[op_base + 1] = .{ .elementwise = .{ .op = .add, .dst = residual_buf, .src0 = q_buf, .src1 = residual_buf, .n = @intCast(elems) } };
        ops[op_base + 2] = .{ .rmsnorm = .{ .dst = norm_buf, .src = residual_buf, .rows = @intCast(case.m), .cols = @intCast(case.n), .eps = 1e-5 } };
        ops[op_base + 3] = .{ .repeat = .{
            .dst = repeat_buf,
            .src = scale_buf,
            .n = @intCast(elems),
            .src_ne = .{ @intCast(case.n), 1, 1, 1 },
            .dst_ne = .{ @intCast(case.n), @intCast(case.m), 1, 1 },
            .src_strides = .{ 1, @intCast(case.n), @intCast(case.n), @intCast(case.n) },
            .dst_strides = .{ 1, @intCast(case.n), @intCast(elems), @intCast(elems) },
        } };
        ops[op_base + 4] = .{ .elementwise = .{ .op = .mul, .dst = out_buf, .src0 = norm_buf, .src1 = repeat_buf, .n = @intCast(elems) } };

        buffer_sizes[1 + slot * 6] = elems;
        buffer_sizes[2 + slot * 6] = elems;
        buffer_sizes[3 + slot * 6] = elems;
        buffer_sizes[4 + slot * 6] = case.n;
        buffer_sizes[5 + slot * 6] = elems;
        buffer_sizes[6 + slot * 6] = elems;
        uploads[1 + slot * 6] = programIo(q_buf, q_out[elem_base .. elem_base + elems]);
        uploads[2 + slot * 6] = programIo(residual_buf, residual[elem_base .. elem_base + elems]);
        uploads[3 + slot * 6] = programIo(norm_buf, norm_out[elem_base .. elem_base + elems]);
        uploads[4 + slot * 6] = programIo(scale_buf, scale[scale_base .. scale_base + case.n]);
        uploads[5 + slot * 6] = programIo(repeat_buf, repeat_out[elem_base .. elem_base + elems]);
        uploads[6 + slot * 6] = programIo(out_buf, staged_out[elem_base .. elem_base + elems]);
        qweights[slot] = .{
            .data = qdata[weight_base .. weight_base + case.k * case.n],
            .scales = qscales[qscale_base .. qscale_base + scale_len],
            .rows = case.k,
            .cols = case.n,
            .block_size = block_size,
        };
    }

    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
        .qweights = &qweights,
    };

    const be = metal.backend();
    var staged_policy = program_mod.CommandStreamPolicy.default();
    staged_policy.fuse_projection_row_chain = false;
    const staged_handle = metal.compileProgramWithCommandPolicy(program, staged_policy) orelse return error.CompileFailed;
    defer be.freeProgram(staged_handle);
    var grouped_policy = program_mod.CommandStreamPolicy.default();
    grouped_policy.fuse_projection_row_chain = true;
    grouped_policy.fuse_projection_row_chain_qmatvec = true;
    const grouped_handle = metal.compileProgramWithCommandPolicy(program, grouped_policy) orelse return error.CompileFailed;
    defer be.freeProgram(grouped_handle);
    const two_phase_policy = program_mod.CommandStreamPolicy.promptProjectionRowChainTwoPhaseCandidate();
    const two_phase_handle = metal.compileProgramWithCommandPolicy(program, two_phase_policy) orelse return error.CompileFailed;
    defer be.freeProgram(two_phase_handle);

    var staged_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var grouped_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    var two_phase_outputs: [n_slots]backend_mod.ProgramIO = undefined;
    for (0..n_slots) |slot| {
        const out_buf: u16 = @intCast(6 + slot * 6);
        const base = slot * elems;
        staged_outputs[slot] = programIo(out_buf, staged_out[base .. base + elems]);
        grouped_outputs[slot] = programIo(out_buf, grouped_out[base .. base + elems]);
        two_phase_outputs[slot] = programIo(out_buf, two_phase_out[base .. base + elems]);
    }

    var staged_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = staged_handle,
        .out = staged_out,
        .output_io = &staged_outputs,
    };
    var grouped_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = grouped_handle,
        .out = grouped_out,
        .output_io = &grouped_outputs,
    };
    var two_phase_bench = ProjectionRowChainMetalBench{
        .be = be,
        .handle = two_phase_handle,
        .out = two_phase_out,
        .output_io = &two_phase_outputs,
    };

    be.executeProgram(staged_handle, &.{}, &staged_outputs);
    be.executeProgram(grouped_handle, &.{}, &grouped_outputs);
    be.executeProgram(two_phase_handle, &.{}, &two_phase_outputs);
    const grouped_max_abs_diff = maxAbsDiff(staged_out, grouped_out);
    const two_phase_max_abs_diff = maxAbsDiff(staged_out, two_phase_out);

    const staged_stats = measure(io, &staged_bench);
    const grouped_stats = measure(io, &grouped_bench);
    const two_phase_stats = measure(io, &two_phase_bench);
    const approx_work = 2.0 * @as(f64, @floatFromInt(n_slots * case.m * case.n * case.k));

    var staged_name_buf: [112]u8 = undefined;
    const staged_name = try std.fmt.bufPrint(&staged_name_buf, "{s} staged", .{case.name});
    try printStats(w, staged_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", staged_stats);

    var grouped_name_buf: [112]u8 = undefined;
    const grouped_name = try std.fmt.bufPrint(&grouped_name_buf, "{s} projection_row_chain_group", .{case.name});
    try printStats(w, grouped_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", grouped_stats);

    var two_phase_name_buf: [128]u8 = undefined;
    const two_phase_name = try std.fmt.bufPrint(&two_phase_name_buf, "{s} projection_row_chain_two_phase_group", .{case.name});
    try printStats(w, two_phase_name, "throughput", approx_work / 1_000_000_000.0, "GFLOP", two_phase_stats);

    var ratio_name_buf: [112]u8 = undefined;
    const ratio_name = try std.fmt.bufPrint(&ratio_name_buf, "{s} projection_row_chain_group", .{case.name});
    try printRatio(w, ratio_name, staged_stats, grouped_stats, grouped_max_abs_diff);

    var two_phase_ratio_name_buf: [144]u8 = undefined;
    const two_phase_ratio_name = try std.fmt.bufPrint(&two_phase_ratio_name_buf, "{s} projection_row_chain_two_phase_group", .{case.name});
    try printRatio(w, two_phase_ratio_name, staged_stats, two_phase_stats, two_phase_max_abs_diff);

    const grouped_commands = try program_mod.buildProgramCommands(alloc, &ops, grouped_policy);
    defer alloc.free(grouped_commands);
    var profile_name_buf: [128]u8 = undefined;
    const profile_name = try std.fmt.bufPrint(&profile_name_buf, "{s} projection_row_chain_group dispatch_profile", .{case.name});
    try printCommandShape(w, profile_name, grouped_commands);
    try printProjectionRowChainRuntimeProfile(w, profile_name, be, grouped_handle, &grouped_outputs);

    const two_phase_commands = try program_mod.buildProgramCommands(alloc, &ops, two_phase_policy);
    defer alloc.free(two_phase_commands);
    var two_phase_profile_name_buf: [144]u8 = undefined;
    const two_phase_profile_name = try std.fmt.bufPrint(&two_phase_profile_name_buf, "{s} projection_row_chain_two_phase_group dispatch_profile", .{case.name});
    try printCommandShape(w, two_phase_profile_name, two_phase_commands);
    try printProjectionRowChainRuntimeProfile(w, two_phase_profile_name, be, two_phase_handle, &two_phase_outputs);
}

fn benchProjectionRowChainMetal(io: std.Io, alloc: std.mem.Allocator, w: *std.Io.Writer, filter: FrontierFilter) !void {
    if (!filter.matchesAny(&.{
        "Metal",
        "qproj",
        "qproj region",
        "qrow",
        "projection_chain",
        "projection_group",
        "projection_group_region",
        "projection_row_chain",
        "qrow region",
        "projection_row_chain_two_phase_group",
    })) return;
    try w.print("\nMetal Projection Row-Chain Command\n", .{});
    try w.print("----------------------------------\n", .{});

    if (@import("builtin").os.tag != .macos or !opts.use_metal) {
        try w.print("  projection_row_chain metal unavailable\n", .{});
        return;
    }

    var metal = internal.backend_metal.MetalBackend.initWithAllocator(alloc) catch |err| switch (err) {
        error.MetalNotAvailable => {
            try w.print("  projection_row_chain metal unavailable\n", .{});
            return;
        },
        else => return err,
    };
    defer metal.deinit();
    metal.setRegionProgramDispatch(true);

    const projection_chain_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj prompt m=32 n=512 k=512", .m = 32, .n = 512, .k = 512 },
        .{ .name = "qproj full-prefill m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj smollm-prompt m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_chain_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj", "projection_chain" })) {
            try benchProjectionChainMetalCase(io, alloc, w, &metal, case);
        }
    }

    const projection_group_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj group full-prefill x4 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj group smollm-prompt x4 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_group_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj group", "projection_group" })) {
            try benchProjectionGroupMetalCase(4, io, alloc, w, &metal, case);
        }
    }

    const projection_group_region_cases = [_]ProjectionRowChainCase{
        .{ .name = "qproj region full-prefill x7 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qproj region smollm-prompt x7 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (projection_group_region_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qproj region", "projection_group_region" })) {
            try benchProjectionGroupMetalCase(7, io, alloc, w, &metal, case);
        }
    }

    const row_chain_group_cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow group full-prefill x4 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow group smollm-prompt x4 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (row_chain_group_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow group", "projection_row_chain_group" })) {
            try benchProjectionRowChainGroupMetalCase(4, io, alloc, w, &metal, case);
        }
    }

    const row_chain_region_cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow region full-prefill x7 m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow region smollm-prompt x7 m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (row_chain_region_cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow region", "projection_row_chain_two_phase_group" })) {
            try benchProjectionRowChainGroupMetalCase(7, io, alloc, w, &metal, case);
        }
    }

    const cases = [_]ProjectionRowChainCase{
        .{ .name = "qrow decode m=1 n=512 k=512", .m = 1, .n = 512, .k = 512 },
        .{ .name = "qrow tiny m=2 n=128 k=128", .m = 2, .n = 128, .k = 128 },
        .{ .name = "qrow prompt m=32 n=512 k=512", .m = 32, .n = 512, .k = 512 },
        .{ .name = "qrow full-prefill m=128 n=512 k=512", .m = 128, .n = 512, .k = 512 },
        .{ .name = "qrow smollm-prompt m=128 n=576 k=576", .m = 128, .n = 576, .k = 576 },
    };
    for (cases) |case| {
        if (filter.matchesAny(&.{ case.name, "qrow", "projection_row_chain" })) {
            try benchProjectionRowChainMetalCase(io, alloc, w, &metal, case);
        }
    }
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
    const filter = FrontierFilter.init();
    if (filter.query) |query| {
        try w.print("filter={s}\n", .{query});
    }

    try benchElementwise(io, alloc, w, filter);
    try benchMatmul(io, alloc, w, filter);
    try benchNorms(io, alloc, w, filter);
    try benchDecodeGraph(io, alloc, w, filter);
    try benchProjectionRowChainMetal(io, alloc, w, filter);

    try w.print("\n", .{});
    writer.interface.flush() catch {};
}
