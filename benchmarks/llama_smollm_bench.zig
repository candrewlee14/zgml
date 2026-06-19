//! SmolLM LLaMA inference benchmark for internal.
//!
//! Measures prompt and decode throughput on the local SmolLM checkpoint using
//! the real `LlamaInferenceSession` path, without tokenizer or stdout noise.
//!
//! Run:
//!   zig build bench-llama-smollm # model-free copy-and-patch stencil probe
//!   ./zig-out/bin/bench-llama-smollm [model.safetensors|model.gguf] [prompt_tokens] [gen_tokens] [repetitions]
//!   ./zig-out/bin/bench-llama-smollm model.gguf 128 200 3 --metal-prefill-device --gate-only
//!   ./zig-out/bin/bench-llama-smollm model.gguf 128 200 3 --metal-prefill-device --metal-decode-region --gate-only
//!   ./zig-out/bin/bench-llama-smollm model.gguf 128 200 3 --metal-prefill-device --metal-prompt-projection-row-chain-candidate --gate-only
//!   ./zig-out/bin/bench-llama-smollm model.gguf 128 200 3 --metal-decode-no-readback
//!   ./zig-out/bin/bench-llama-smollm ignored 128 1 1 --stencil-only

const std = @import("std");
const internal = @import("zgml_internal");
const opts = @import("zgml_options");

const CpuBackend = internal.backend_cpu.CpuBackend;
const MetalBackend = internal.backend_metal.MetalBackend;
const StencilBackend = internal.backend_stencil.StencilBackend;
const Backend = internal.backend.Backend;
const program_mod = internal.backend_program;
const is_macos = @import("builtin").os.tag == .macos;

const config = internal.llama_inference.LlamaConfig{
    .vocab_size = 49152,
    .d_model = 576,
    .n_heads = 9,
    .n_kv_heads = 3,
    .d_ff = 1536,
    .n_layers = 30,
    .max_seq_len = 2048,
    .rope_base = 10000.0,
    .rms_norm_eps = 1e-5,
    .tied_lm_head = true,
};

const Session = internal.llama_inference.LlamaInferenceSession(f32, config);
const expected_p128_decode_stencil_hash: u64 = 14405191909906507341;
const expected_p128_prefill_stencil_hash: u64 = 17558208047327870709;
const StencilPhase = enum { decode, prompt };

const BenchConfig = struct {
    model_path: []const u8,
    prompt_tokens: usize,
    gen_tokens: usize,
    repetitions: usize,
};

const JsonRow = union(enum) {
    bench: struct { prompt_tok_s: ?f64, decode_tok_s: ?f64 },
    stencil: struct { phase: []const u8 },
};

fn writeSemanticShapeJsonFields(jw: *std.json.Stringify, shape: anytype) !void {
    try jw.objectField("semantic_stage_count");
    try jw.write(shape.semantic_stage_count);
    try jw.objectField("semantic_token_count");
    try jw.write(shape.semantic_token_count);
    try jw.objectField("semantic_layers");
    try jw.write(shape.semantic_layers);
    try jw.objectField("semantic_heads");
    try jw.write(shape.semantic_heads);
    try jw.objectField("semantic_kv_heads");
    try jw.write(shape.semantic_kv_heads);
    try jw.objectField("semantic_layer_stage_count");
    try jw.write(shape.semantic_layer_stage_count);
    try jw.objectField("semantic_terminal_stage_count");
    try jw.write(shape.semantic_terminal_stage_count);
    try jw.objectField("semantic_runtime_patch_holes");
    try jw.write(shape.semantic_runtime_patch_holes);
    try jw.objectField("semantic_runtime_patch_cache_write_pos_holes");
    try jw.write(shape.semantic_runtime_patch_cache_write_pos_holes);
    try jw.objectField("semantic_runtime_patch_attention_seq_kv_holes");
    try jw.write(shape.semantic_runtime_patch_attention_seq_kv_holes);
}

fn writeProfileJson(
    writer: anytype,
    label: []const u8,
    row: JsonRow,
    profile: internal.profile.RuntimeProfile,
    shape: anytype,
) !void {
    try writer.writeAll(switch (row) {
        .bench => "ZGML_BENCH_JSON ",
        .stencil => "ZGML_STENCIL_JSON ",
    });
    var jw: std.json.Stringify = .{ .writer = writer };
    try jw.beginObject();
    try jw.objectField("label");
    try jw.write(label);
    switch (row) {
        .bench => |bench| {
            try jw.objectField("prompt_tok_s");
            try jw.write(bench.prompt_tok_s);
            try jw.objectField("decode_tok_s");
            try jw.write(bench.decode_tok_s);
        },
        .stencil => |stencil| {
            try jw.objectField("phase");
            try jw.write(stencil.phase);
        },
    }
    try writeSemanticShapeJsonFields(&jw, shape);
    try internal.profile.writeRuntimeProfileJsonFields(profile, &jw);
    try jw.endObject();
    try writer.writeByte('\n');
    writer.flush() catch {};
}

fn requireStencilEvidence(comptime phase: StencilPhase, prompt_tokens: usize, profile: internal.profile.RuntimeProfile, shape: anytype) !void {
    const patch = profile.runtime_patch_shape;
    if (profile.runtime_patch_call_count != 0) return error.UnexpectedStencilRuntimeCalls;
    if (patch.runtime_patch_holes != shape.semantic_runtime_patch_holes or
        patch.runtime_patch_cache_write_pos_holes != shape.semantic_runtime_patch_cache_write_pos_holes or
        patch.runtime_patch_attention_seq_kv_holes != shape.semantic_runtime_patch_attention_seq_kv_holes) return error.RuntimePatchShapeMismatch;
    if (prompt_tokens == 128) {
        const expected_hash = switch (phase) {
            .decode => expected_p128_decode_stencil_hash,
            .prompt => expected_p128_prefill_stencil_hash,
        };
        if (patch.runtime_patch_stencil_hash != expected_hash) return error.StencilHashMismatch;
    }
}

fn decodeToken(decode: anytype, token_id: usize, discard_logits: bool) !void {
    if (discard_logits) {
        try decode.advance(token_id);
    } else {
        _ = try decode.step(token_id);
    }
}

fn runVariant(
    label: []const u8,
    maybe_backend: ?Backend,
    quantized: bool,
    quant_kv: bool,
    cfg: BenchConfig,
    writer: anytype,
    io: std.Io,
    alloc: std.mem.Allocator,
) !void {
    const total_tokens = std.math.add(usize, cfg.prompt_tokens, cfg.gen_tokens) catch return error.SequenceTooLong;
    if (total_tokens > config.max_seq_len) return error.SequenceTooLong;

    var session = if (maybe_backend) |backend|
        try Session.initWithBackend(alloc, backend)
    else
        try Session.init(alloc);
    defer session.deinit();

    try session.load(io, cfg.model_path);
    if (quantized and !std.ascii.endsWithIgnoreCase(cfg.model_path, ".gguf")) try session.quantize();
    if (quant_kv) try session.quantizeKV();

    // Pre-build the prompt token buffer once. The session takes it verbatim.
    const prompt = try alloc.alloc(usize, cfg.prompt_tokens);
    defer alloc.free(prompt);
    for (prompt, 0..) |*t, i| t.* = (i + 1) % config.vocab_size;

    // Warm up the real decode path and prefill path once each to settle
    // kernel/backend selection and amortize plan-build cost.
    _ = try session.step(0);
    session.reset();
    _ = try session.prefill(prompt);
    session.reset();

    var prompt_total_ns: u128 = 0;
    var gen_total_ns: u128 = 0;

    for (0..cfg.repetitions) |_| {
        session.reset();
        const prompt_start = std.Io.Clock.awake.now(io).nanoseconds;
        _ = try session.prefill(prompt);
        const prompt_end = std.Io.Clock.awake.now(io).nanoseconds;
        prompt_total_ns += @intCast(prompt_end - prompt_start);

        const gen_start = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..cfg.gen_tokens) |i| {
            _ = try session.step((cfg.prompt_tokens + i + 1) % config.vocab_size);
        }
        const gen_end = std.Io.Clock.awake.now(io).nanoseconds;
        gen_total_ns += @intCast(gen_end - gen_start);
    }

    const prompt_ns = @as(f64, @floatFromInt(prompt_total_ns));
    const gen_ns = @as(f64, @floatFromInt(gen_total_ns));
    const prompt_tok_s = @as(f64, @floatFromInt(cfg.prompt_tokens * cfg.repetitions)) / (prompt_ns / 1_000_000_000.0);
    const gen_tok_s = @as(f64, @floatFromInt(cfg.gen_tokens * cfg.repetitions)) / (gen_ns / 1_000_000_000.0);
    const prompt_avg_ms = prompt_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;
    const gen_avg_ms = gen_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;

    try writer.print(
        "  {s}: prompt {d:>7.1} tok/s ({d:>6.2} ms avg)  decode {d:>7.1} tok/s ({d:>6.2} ms avg)\n",
        .{ label, prompt_tok_s, prompt_avg_ms, gen_tok_s, gen_avg_ms },
    );
    writer.flush() catch {};
}

/// Run decode-only benchmark through the reusable device decode path.
fn runDeviceVariant(
    label: []const u8,
    be: Backend,
    cfg: BenchConfig,
    writer: anytype,
    io: std.Io,
    alloc: std.mem.Allocator,
    discard_logits: bool,
) !void {
    // Build session without backend — graph captures all ops for device compilation.
    var session = try Session.init(alloc);
    defer session.deinit();
    try session.load(io, cfg.model_path);

    // Logits host buffer.
    const logits_buf = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_buf);

    var decode_program = try session.compileDeviceDecodeProgram(be, alloc);
    defer decode_program.deinit();
    var decode = try decode_program.bind(&session, logits_buf);
    defer decode.deinit();

    // Warm up the same prompt-seeded decode path measured below.
    const total_tokens = std.math.add(usize, cfg.prompt_tokens, cfg.gen_tokens) catch return error.SequenceTooLong;
    if (total_tokens > config.max_seq_len) return error.SequenceTooLong;
    for (0..total_tokens) |i| {
        try decodeToken(&decode, (i + 1) % config.vocab_size, discard_logits);
    }
    session.reset();

    // Benchmark decode after the requested prompt length, matching llama-bench tg.
    var gen_total_ns: u128 = 0;
    var gen_profile = internal.profile.RuntimeProfile{};
    for (0..cfg.repetitions) |_| {
        session.reset();
        decode.resetRuntimeProfile();
        for (0..cfg.prompt_tokens) |i| {
            try decodeToken(&decode, (i + 1) % config.vocab_size, discard_logits);
        }
        decode.resetRuntimeProfile();

        const gen_start = std.Io.Clock.awake.now(io).nanoseconds;
        for (0..cfg.gen_tokens) |i| {
            try decodeToken(&decode, (cfg.prompt_tokens + i + 1) % config.vocab_size, discard_logits);
        }
        const gen_end = std.Io.Clock.awake.now(io).nanoseconds;
        gen_total_ns += @intCast(gen_end - gen_start);
        decode.addRuntimeProfileTo(&gen_profile);
    }

    const gen_ns = @as(f64, @floatFromInt(gen_total_ns));
    const gen_tok_s = @as(f64, @floatFromInt(cfg.gen_tokens * cfg.repetitions)) / (gen_ns / 1_000_000_000.0);
    const gen_avg_ms = gen_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;

    try writer.print(
        "  {s}: prompt {s:>7} tok/s ({s:>6} ms avg)  decode {d:>7.1} tok/s ({d:>6.2} ms avg)\n",
        .{ label, "  —  ", " —  ", gen_tok_s, gen_avg_ms },
    );
    writer.flush() catch {};

    try writeProfileJson(writer, label, .{ .bench = .{ .prompt_tok_s = null, .decode_tok_s = gen_tok_s } }, gen_profile, decode.semanticShape());
}

/// Run prompt/prefill through the reusable device prefill path.
fn runDevicePrefillVariant(
    label: []const u8,
    be: Backend,
    cfg: BenchConfig,
    writer: anytype,
    io: std.Io,
    alloc: std.mem.Allocator,
) !void {
    if (cfg.prompt_tokens == 0 or cfg.prompt_tokens > config.max_seq_len) return error.InvalidPromptLength;

    var session = try Session.init(alloc);
    defer session.deinit();
    try session.load(io, cfg.model_path);

    const prompt = try alloc.alloc(usize, cfg.prompt_tokens);
    defer alloc.free(prompt);
    for (prompt, 0..) |*t, i| t.* = (i + 1) % config.vocab_size;

    const logits_buf = try alloc.alloc(f32, config.vocab_size);
    defer alloc.free(logits_buf);

    var prefill_program = try session.compileDevicePrefillProgram(be, alloc, cfg.prompt_tokens);
    defer prefill_program.deinit();
    var prefill = try prefill_program.bind(&session, logits_buf);
    defer prefill.deinit();

    _ = try prefill.prefill(prompt);
    session.reset();
    prefill.resetRuntimeProfile();

    var prompt_total_ns: u128 = 0;
    for (0..cfg.repetitions) |_| {
        session.reset();
        const prompt_start = std.Io.Clock.awake.now(io).nanoseconds;
        _ = try prefill.prefill(prompt);
        const prompt_end = std.Io.Clock.awake.now(io).nanoseconds;
        prompt_total_ns += @intCast(prompt_end - prompt_start);
    }

    const prompt_ns = @as(f64, @floatFromInt(prompt_total_ns));
    const prompt_tok_s = @as(f64, @floatFromInt(cfg.prompt_tokens * cfg.repetitions)) / (prompt_ns / 1_000_000_000.0);
    const prompt_avg_ms = prompt_ns / @as(f64, @floatFromInt(cfg.repetitions)) / 1_000_000.0;

    try writer.print(
        "  {s}: prompt {d:>7.1} tok/s ({d:>6.2} ms avg)  decode {s:>7} tok/s ({s:>6} ms avg)\n",
        .{ label, prompt_tok_s, prompt_avg_ms, "  —  ", " —  " },
    );
    writer.flush() catch {};

    var prefill_profile = internal.profile.RuntimeProfile{};
    prefill.addRuntimeProfileTo(&prefill_profile);
    try writeProfileJson(writer, label, .{ .bench = .{ .prompt_tok_s = prompt_tok_s, .decode_tok_s = null } }, prefill_profile, prefill.semanticShape());
}

fn runStencilProbe(
    cfg: BenchConfig,
    writer: anytype,
    alloc: std.mem.Allocator,
    debug_row_chain: bool,
) !void {
    if (cfg.prompt_tokens == 0 or cfg.prompt_tokens > config.max_seq_len) return error.InvalidPromptLength;

    var session = try Session.init(alloc);
    defer session.deinit();

    var stencil_backend = StencilBackend.init(internal.backend.Capabilities.metal);
    var decode = try session.compileDeviceDecodeProgram(stencil_backend.backend(), alloc);
    defer decode.deinit();
    if (debug_row_chain) {
        const debug = internal.backend_stencil.firstProjectionRowChainFrontierDebug(decode.program.handle);
        try writer.print(
            "ZGML_ROW_CHAIN_DEBUG phase=decode reason={s} command_count={d} first_kind={s} first_projection={d}/{s}/{s}/{s}/op_start={d}/op_count={d}/sidecars={d}/ops={s},{s},{s},{s},{s},{s} command_index={d} op_start={d} row_start={d} q_m={d} q_n={d} q_dst={d} ew_dst={d} ew_src0={d} ew_src1={d} rms_src={d} rms_dst={d}\n",
            .{
                @tagName(debug.reason),
                debug.command_count,
                @tagName(debug.first_command_kind),
                debug.first_projection_command_index,
                @tagName(debug.first_projection_prev_kind),
                @tagName(debug.first_projection_kind),
                @tagName(debug.first_projection_next_kind),
                debug.first_projection_op_start,
                debug.first_projection_op_count,
                debug.first_projection_sidecars,
                @tagName(debug.first_projection_op_tags[0]),
                @tagName(debug.first_projection_op_tags[1]),
                @tagName(debug.first_projection_op_tags[2]),
                @tagName(debug.first_projection_op_tags[3]),
                @tagName(debug.first_projection_op_tags[4]),
                @tagName(debug.first_projection_op_tags[5]),
                debug.projection_command_index,
                debug.projection_op_start,
                debug.row_op_start,
                debug.q_m,
                debug.q_n,
                debug.q_dst,
                debug.elementwise_dst,
                debug.elementwise_src0,
                debug.elementwise_src1,
                debug.rms_src,
                debug.rms_dst,
            },
        );
        const projection_debug = internal.backend_stencil.firstProjectionElementwiseChainDebug(decode.program.handle);
        try writer.print(
            "ZGML_PROJECTION_CHAIN_DEBUG phase=decode reason={s} command_count={d} command_index={d} prev={s} kind={s} next={s} op_start={d} op_count={d} q_m={d} q_n={d} q_dst={d} q_input={d} weight_idx={d} ew_op={s} ew_dst={d} ew_src0={d} ew_src1={d} ew_n={d} primary_external={any}\n",
            .{
                @tagName(projection_debug.reason),
                projection_debug.command_count,
                projection_debug.command_index,
                @tagName(projection_debug.prev_kind),
                @tagName(projection_debug.kind),
                @tagName(projection_debug.next_kind),
                projection_debug.op_start,
                projection_debug.op_count,
                projection_debug.q_m,
                projection_debug.q_n,
                projection_debug.q_dst,
                projection_debug.q_input,
                projection_debug.weight_idx,
                @tagName(projection_debug.elementwise_op),
                projection_debug.elementwise_dst,
                projection_debug.elementwise_src0,
                projection_debug.elementwise_src1,
                projection_debug.elementwise_n,
                projection_debug.primary_has_external_users,
            },
        );
    }
    var decode_profile = internal.profile.RuntimeProfile{};
    decode.addRuntimeProfileTo(&decode_profile);
    const decode_shape = decode.semanticShape();
    try requireStencilEvidence(.decode, cfg.prompt_tokens, decode_profile, decode_shape);
    try writeProfileJson(writer, "metal stencil decode", .{ .stencil = .{ .phase = "decode" } }, decode_profile, decode_shape);

    var prefill = try session.compileDevicePrefillProgram(stencil_backend.backend(), alloc, cfg.prompt_tokens);
    defer prefill.deinit();
    if (debug_row_chain) {
        const debug = internal.backend_stencil.firstProjectionRowChainFrontierDebug(prefill.program.handle);
        try writer.print(
            "ZGML_ROW_CHAIN_DEBUG phase=prompt reason={s} command_count={d} first_kind={s} first_projection={d}/{s}/{s}/{s}/op_start={d}/op_count={d}/sidecars={d}/ops={s},{s},{s},{s},{s},{s} command_index={d} op_start={d} row_start={d} q_m={d} q_n={d} q_dst={d} ew_dst={d} ew_src0={d} ew_src1={d} rms_src={d} rms_dst={d}\n",
            .{
                @tagName(debug.reason),
                debug.command_count,
                @tagName(debug.first_command_kind),
                debug.first_projection_command_index,
                @tagName(debug.first_projection_prev_kind),
                @tagName(debug.first_projection_kind),
                @tagName(debug.first_projection_next_kind),
                debug.first_projection_op_start,
                debug.first_projection_op_count,
                debug.first_projection_sidecars,
                @tagName(debug.first_projection_op_tags[0]),
                @tagName(debug.first_projection_op_tags[1]),
                @tagName(debug.first_projection_op_tags[2]),
                @tagName(debug.first_projection_op_tags[3]),
                @tagName(debug.first_projection_op_tags[4]),
                @tagName(debug.first_projection_op_tags[5]),
                debug.projection_command_index,
                debug.projection_op_start,
                debug.row_op_start,
                debug.q_m,
                debug.q_n,
                debug.q_dst,
                debug.elementwise_dst,
                debug.elementwise_src0,
                debug.elementwise_src1,
                debug.rms_src,
                debug.rms_dst,
            },
        );
        const projection_debug = internal.backend_stencil.firstProjectionElementwiseChainDebug(prefill.program.handle);
        try writer.print(
            "ZGML_PROJECTION_CHAIN_DEBUG phase=prompt reason={s} command_count={d} command_index={d} prev={s} kind={s} next={s} op_start={d} op_count={d} q_m={d} q_n={d} q_dst={d} q_input={d} weight_idx={d} ew_op={s} ew_dst={d} ew_src0={d} ew_src1={d} ew_n={d} primary_external={any}\n",
            .{
                @tagName(projection_debug.reason),
                projection_debug.command_count,
                projection_debug.command_index,
                @tagName(projection_debug.prev_kind),
                @tagName(projection_debug.kind),
                @tagName(projection_debug.next_kind),
                projection_debug.op_start,
                projection_debug.op_count,
                projection_debug.q_m,
                projection_debug.q_n,
                projection_debug.q_dst,
                projection_debug.q_input,
                projection_debug.weight_idx,
                @tagName(projection_debug.elementwise_op),
                projection_debug.elementwise_dst,
                projection_debug.elementwise_src0,
                projection_debug.elementwise_src1,
                projection_debug.elementwise_n,
                projection_debug.primary_has_external_users,
            },
        );
    }
    var prefill_profile = internal.profile.RuntimeProfile{};
    prefill.addRuntimeProfileTo(&prefill_profile);
    const prefill_shape = prefill.semanticShape();
    try requireStencilEvidence(.prompt, cfg.prompt_tokens, prefill_profile, prefill_shape);
    try writeProfileJson(writer, "metal stencil prefill", .{ .stencil = .{ .phase = "prompt" } }, prefill_profile, prefill_shape);
}

fn parseArgOrDefault(args: []const []const u8, idx: usize, default: usize) !usize {
    if (idx >= args.len) return default;
    return try std.fmt.parseInt(usize, args[idx], 10);
}

fn hasFlag(args: []const []const u8, flag: []const u8) bool {
    if (args.len <= 5) return false;
    for (args[5..]) |arg| {
        if (std.mem.eql(u8, arg, flag)) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const stdout_file = std.Io.File.stdout();
    var stdout_buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(io, &stdout_buf);

    const cfg = BenchConfig{
        .model_path = if (args.len > 1) args[1] else "data/smollm/model.safetensors",
        .prompt_tokens = try parseArgOrDefault(args, 2, 4),
        .gen_tokens = try parseArgOrDefault(args, 3, 200),
        .repetitions = try parseArgOrDefault(args, 4, 3),
    };
    const run_metal_prefill_device = hasFlag(args, "--metal-prefill-device");
    const run_metal_decode_region = hasFlag(args, "--metal-decode-region");
    const run_metal_decode_no_readback = hasFlag(args, "--metal-decode-no-readback");
    const run_metal_prompt_projection_row_chain_command_candidate = hasFlag(args, "--metal-prompt-projection-row-chain-command-candidate");
    const run_metal_prompt_projection_row_chain_candidate = hasFlag(args, "--metal-prompt-projection-row-chain-candidate");
    const stencil_only = hasFlag(args, "--stencil-only");
    const debug_row_chain = hasFlag(args, "--debug-row-chain");
    const gate_only = hasFlag(args, "--gate-only");
    const model_is_gguf = std.ascii.endsWithIgnoreCase(cfg.model_path, ".gguf");

    try stdout.interface.print("\nSmolLM LLaMA Benchmark — zgml\n", .{});
    try stdout.interface.print("================================\n", .{});
    try stdout.interface.print("  model={s}\n", .{cfg.model_path});
    try stdout.interface.print("  prompt={d}, gen={d}, reps={d}\n\n", .{ cfg.prompt_tokens, cfg.gen_tokens, cfg.repetitions });
    stdout.interface.flush() catch {};

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    if (stencil_only) {
        try runStencilProbe(cfg, &stdout.interface, alloc, debug_row_chain);
        try stdout.interface.writeByte('\n');
        stdout.interface.flush() catch {};
        return;
    }

    if (!gate_only) {
        try runVariant(if (model_is_gguf) "default gguf    " else "default f32      ", null, false, false, cfg, &stdout.interface, io, alloc);

        var cpu_backend = CpuBackend{};
        try runVariant(if (model_is_gguf) "cpu-backend gguf" else "cpu-backend f32  ", cpu_backend.backend(), false, false, cfg, &stdout.interface, io, alloc);
        if (!model_is_gguf) {
            try runVariant("default int8     ", null, true, false, cfg, &stdout.interface, io, alloc);
            try runVariant("cpu-backend i8   ", cpu_backend.backend(), true, false, cfg, &stdout.interface, io, alloc);
            try runVariant("i8 + kv-i8       ", null, true, true, cfg, &stdout.interface, io, alloc);
            try runVariant("kv-i8 only       ", null, false, true, cfg, &stdout.interface, io, alloc);
        } else {
            try runVariant("gguf + kv-i8    ", null, false, true, cfg, &stdout.interface, io, alloc);
        }
    }

    if (opts.use_metal and is_macos) metal: {
        var metal_be = MetalBackend.init() catch |err| {
            try stdout.interface.print("  metal init failed: {}\n", .{err});
            break :metal;
        };
        defer metal_be.deinit();
        const metal_prefill_label = if (run_metal_prompt_projection_row_chain_candidate)
            "metal scheduled prefill projection-row-chain candidate"
        else if (run_metal_prompt_projection_row_chain_command_candidate)
            "metal scheduled prefill projection-row-chain command candidate"
        else
            "metal scheduled prefill";
        if (run_metal_prompt_projection_row_chain_candidate) {
            metal_be.setCommandStreamPolicy(program_mod.CommandStreamPolicy.promptProjectionRowChainCandidate());
        } else if (run_metal_prompt_projection_row_chain_command_candidate) {
            var command_policy = program_mod.CommandStreamPolicy.default();
            command_policy.fuse_projection_row_chain = true;
            command_policy.fuse_projection_row_chain_qmatvec = false;
            command_policy.fuse_projection_row_chain_single_dispatch = false;
            command_policy.min_projection_row_chain_rows = 8;
            metal_be.setCommandStreamPolicy(command_policy);
        }
        if (!gate_only) {
            try runVariant(if (model_is_gguf) "metal gguf      " else "metal f32        ", metal_be.backend(), false, false, cfg, &stdout.interface, io, alloc);
            if (!model_is_gguf) try runVariant("metal int8       ", metal_be.backend(), true, false, cfg, &stdout.interface, io, alloc);
        }
        if (run_metal_prefill_device and model_is_gguf) {
            metal_be.setRegionProgramDispatch(true);
            try runDevicePrefillVariant(metal_prefill_label, metal_be.backend(), cfg, &stdout.interface, io, alloc);
            metal_be.setRegionProgramDispatch(false);
        }
        if (!gate_only or !run_metal_decode_region) {
            try runDeviceVariant("metal cpu-fallback decode", metal_be.backend(), cfg, &stdout.interface, io, alloc, false);
        }
        if (run_metal_decode_region and model_is_gguf) {
            metal_be.setRegionProgramDispatch(true);
            try runDeviceVariant("metal region decode", metal_be.backend(), cfg, &stdout.interface, io, alloc, false);
            if (run_metal_decode_no_readback) {
                try runDeviceVariant("metal region decode no-readback", metal_be.backend(), cfg, &stdout.interface, io, alloc, true);
            }
            metal_be.setRegionProgramDispatch(false);
        }
    }

    try stdout.interface.writeByte('\n');
    stdout.interface.flush() catch {};
}
