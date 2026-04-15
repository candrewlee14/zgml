//! Metal inference benchmark: compare CPU vs Metal backend inference.
//!
//! Measures tok/s and us/tok for CPU f32, CPU int8, and Metal int8
//! across small and medium GPT configs.
//!
//! Run with: zig build bench-metal-inference

const std = @import("std");
const zgml = @import("zgml");
const GPTConfig = zgml.models.GPTConfig;
const MetalBackend = zgml.backend_metal.MetalBackend;
const CpuBackend = zgml.backend_cpu.CpuBackend;
const Backend = zgml.backend.Backend;

const WARMUP_TOKENS = 1;
const TIMED_TOKENS = 32;

fn diagnoseGraph(comptime config: GPTConfig, writer: anytype) !void {
    const Session = zgml.inference.InferenceSession(f32, config);
    const Op = zgml.Op;
    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var session = try Session.init(alloc);
    defer session.deinit();
    try session.quantize();

    const steps = session.plan.graph.forward_execution_steps.items;
    const nodes = session.plan.graph.nodes.items[0..session.plan.graph.forward_node_count];

    // GPU-supported ops (must match metal.zig deviceComputeF32)
    const gpu_ops = [_]Op{ .add, .mul, .neg, .exp, .sqrt, .recip, .gelu, .sum, .max, .repeat, .slice_assign };
    const structural = [_]Op{ .none, .view, .as_strided, .reshape, .transpose, .permute, .broadcast_to };

    const fused_mod = @import("zgml").backend; // just for type access
    _ = fused_mod;
    var n_fusion: u32 = 0;
    var n_fused_elem: u32 = 0;
    var n_fused_softmax: u32 = 0;
    var n_fused_layernorm: u32 = 0;
    var n_fused_other: u32 = 0;
    var n_node: u32 = 0;
    var n_matmul: u32 = 0;
    var n_gpu: u32 = 0;
    var n_structural: u32 = 0;
    var n_cpu_fallback: u32 = 0;

    for (steps) |step| {
        switch (step) {
            .fusion => |idx| {
                n_fusion += 1;
                const kind = session.plan.graph.fused_chains.items[idx].kind();
                switch (kind) {
                    .elementwise_chain => n_fused_elem += 1,
                    .softmax => n_fused_softmax += 1,
                    .layer_norm => n_fused_layernorm += 1,
                    else => n_fused_other += 1,
                }
            },
            .node => |node| {
                n_node += 1;
                const op = node.opTag();
                if (op == .matmul) { n_matmul += 1; continue; }
                for (structural) |s| { if (op == s) { n_structural += 1; break; } } else {
                    var found = false;
                    for (gpu_ops) |g| { if (op == g) { found = true; break; } }
                    if (found) n_gpu += 1 else n_cpu_fallback += 1;
                }
            },
        }
    }

    try writer.print("  Graph: {d} steps ({d} fused, {d} nodes)\n", .{ steps.len, n_fusion, n_node });
    try writer.print("  Nodes: {d} total raw, {d} forward\n", .{ session.plan.graph.nodes.items.len, nodes.len });
    try writer.print("  Device: {d} matmul, {d} gpu-compute, {d} structural(skip)\n", .{ n_matmul, n_gpu, n_structural });
    try writer.print("  Fused:  {d} total ({d} elem-chain, {d} softmax, {d} layernorm, {d} other)\n", .{ n_fusion, n_fused_elem, n_fused_softmax, n_fused_layernorm, n_fused_other });
    try writer.print("  CPU fallback: {d} ops", .{n_cpu_fallback});
    if (n_cpu_fallback > 0) {
        try writer.print(" [", .{});
        for (steps) |step| {
            switch (step) {
                .node => |node| {
                    const op = node.opTag();
                    if (op == .matmul) continue;
                    for (structural) |s| { if (op == s) break; } else {
                        var found = false;
                        for (gpu_ops) |g| { if (op == g) { found = true; break; } }
                        if (!found) try writer.print("{s} ", .{op.symbol()});
                    }
                },
                else => {},
            }
        }
        try writer.print("]", .{});
    }
    try writer.print("\n\n", .{});
}

fn runBenchmark(
    comptime label: []const u8,
    comptime config: GPTConfig,
    quantized: bool,
    backend: ?Backend,
    writer: anytype,
    io: std.Io,
) !void {
    const Session = zgml.inference.InferenceSession(f32, config);

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var session = try Session.initWithBackend(alloc, backend);
    defer session.deinit();

    if (quantized) try session.quantize();

    // Warm up.
    for (0..WARMUP_TOKENS) |i| {
        _ = try session.step(i % config.vocab_size);
    }
    session.reset();

    const t_start = std.Io.Clock.awake.now(io).nanoseconds;
    for (0..TIMED_TOKENS) |i| {
        _ = try session.step(i % config.vocab_size);
    }
    const t_end = std.Io.Clock.awake.now(io).nanoseconds;

    const elapsed_ns: f64 = @floatFromInt(t_end - t_start);
    const elapsed_ms = elapsed_ns / 1_000_000.0;
    const tok_per_sec = @as(f64, @floatFromInt(TIMED_TOKENS)) / (elapsed_ns / 1_000_000_000.0);
    const us_per_tok = elapsed_ns / 1000.0 / @as(f64, @floatFromInt(TIMED_TOKENS));

    try writer.print("  {s}: {d:>8.1} tok/s  {d:>7.1} us/tok  {d:>6.1}ms total\n", .{
        label,
        tok_per_sec,
        us_per_tok,
        elapsed_ms,
    });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var w = stdout_file.writer(io, &stdout_buf);

    try w.interface.print("\nMetal Inference Benchmark — CPU vs Metal\n", .{});
    try w.interface.print("==========================================\n", .{});
    try w.interface.print("  warmup={d}, timed={d} tokens\n\n", .{ WARMUP_TOKENS, TIMED_TOKENS });

    // Try to init Metal backend.
    var maybe_metal: ?MetalBackend = MetalBackend.init() catch |err| blk: {
        try w.interface.print("  Metal init failed: {} — skipping Metal tests\n\n", .{err});
        break :blk null;
    };
    defer if (maybe_metal) |*mb| mb.deinit();

    const metal: ?Backend = if (maybe_metal) |*mb| mb.backend() else null;


    const small = GPTConfig{
        .vocab_size = 256,
        .d_model = 64,
        .n_heads = 4,
        .d_ff = 256,
        .n_layers = 4,
        .max_seq_len = 128,
    };

    const medium = GPTConfig{
        .vocab_size = 512,
        .d_model = 128,
        .n_heads = 8,
        .d_ff = 512,
        .n_layers = 6,
        .max_seq_len = 256,
    };

    // GPT-2 124M scale (d=768, 12 layers, 12 heads).
    const gpt2 = GPTConfig{
        .vocab_size = 4096, // reduced from 50257 to keep memory reasonable
        .d_model = 768,
        .n_heads = 12,
        .d_ff = 3072,
        .n_layers = 12,
        .max_seq_len = 256,
    };

    // --- Small ---
    try w.interface.print("Small (d=64, L=4, H=4):\n", .{});
    try runBenchmark("CPU f32   ", small, false, null, &w.interface, io);
    try runBenchmark("CPU int8  ", small, true, null, &w.interface, io);
    if (metal) |be| {
        try runBenchmark("Metal int8", small, true, be, &w.interface, io);
    }

    // --- Medium ---
    try w.interface.print("\nMedium (d=128, L=6, H=8):\n", .{});
    try runBenchmark("CPU f32   ", medium, false, null, &w.interface, io);
    try runBenchmark("CPU int8  ", medium, true, null, &w.interface, io);
    if (metal) |be| {
        try runBenchmark("Metal int8", medium, true, be, &w.interface, io);
    }

    // --- GPT-2 scale ---
    try w.interface.print("\nGPT-2 scale (d=768, L=12, H=12):\n", .{});
    try diagnoseGraph(gpt2, &w.interface);
    try runBenchmark("CPU f32   ", gpt2, false, null, &w.interface, io);
    try runBenchmark("CPU int8  ", gpt2, true, null, &w.interface, io);
    if (metal) |be| {
        try runBenchmark("Metal int8", gpt2, true, be, &w.interface, io);
    }

    try w.interface.print("\n", .{});
    w.interface.flush() catch {};
}
