//! Device-backed LLaMA prefill execution.
//!
//! This is a small bridge from the pure stage plan to the current graph
//! runtime: build one fixed-width prefill graph, compile it once for a backend,
//! then patch only runtime bindings before each execution.

const std = @import("std");

const backend_mod = @import("../backend.zig");
const device_inference = @import("../device_inference.zig");
const llama_inference = @import("../llama_inference.zig");
const LlamaConfig = @import("../models/llama.zig").LlamaConfig;
const Tensor = @import("../tensor.zig").Tensor;
const profile = @import("../profile.zig");

pub fn LlamaDevicePrefill(comptime T: type, comptime config: LlamaConfig) type {
    const Session = llama_inference.LlamaInferenceSession(T, config);
    const Plan = llama_inference.LlamaInferencePlan(T, config);
    const Device = device_inference.DeviceInference(T);
    const TensorT = Tensor(T);
    const d_model = config.d_model;
    const d_head = config.d_model / config.n_heads;
    const max_seq = config.max_seq_len;

    return struct {
        const Self = @This();

        session: *Session,
        plan: *Plan,
        device: Device,
        logits_buf: []T,
        chunk_tokens: usize,

        pub const InitOptions = struct {
            session: *Session,
            backend: backend_mod.Backend,
            alloc: std.mem.Allocator,
            chunk_tokens: usize,
            logits_buf: []T,
        };

        pub fn init(opts: InitOptions) !Self {
            if (opts.chunk_tokens == 0 or opts.chunk_tokens > max_seq) return error.InvalidPrefillLength;
            if (opts.logits_buf.len < config.vocab_size) return error.OutputBufferTooSmall;

            const plan = try opts.session.ensurePrefillPlan(opts.chunk_tokens);

            const input_tensors = try opts.alloc.alloc(*const TensorT, 2 + config.n_layers);
            defer opts.alloc.free(input_tensors);
            input_tensors[0] = plan.token_input;
            input_tensors[1] = plan.attn_mask;
            for (plan.trace.layers, 0..) |lt, l| input_tensors[2 + l] = lt.rope;

            const last_logits = plan.trace.logits.sliceColumns(opts.chunk_tokens - 1, opts.chunk_tokens);
            const device = try Device.init(.{
                .graph = &plan.graph,
                .be = opts.backend,
                .alloc = opts.alloc,
                .input_tensors = input_tensors,
                .output_tensor = last_logits,
                .output_host_buf = opts.logits_buf.ptr,
                .output_len = config.vocab_size,
                .quant_weights = plan.quant_weights,
                .quant_map = &plan.quant_map,
            });

            return .{
                .session = opts.session,
                .plan = plan,
                .device = device,
                .logits_buf = opts.logits_buf[0..config.vocab_size],
                .chunk_tokens = opts.chunk_tokens,
            };
        }

        pub fn deinit(self: *Self) void {
            self.device.deinit();
        }

        /// Execute one fixed-size prefill window at the session's current
        /// position and advance the session position.
        pub fn prefill(self: *Self, tokens: []const usize) ![]const T {
            if (self.session.pos + tokens.len > max_seq) return error.SequenceTooLong;
            const logits = try self.executeAt(tokens, self.session.pos);
            self.session.pos += tokens.len;
            return logits;
        }

        /// Execute one fixed-size prefill window at an explicit position.
        /// This does not update `session.pos`, which keeps benchmarking and
        /// replay code deterministic.
        pub fn executeAt(self: *Self, tokens: []const usize, position: usize) ![]const T {
            if (tokens.len != self.chunk_tokens) return error.InvalidPrefillLength;
            if (position + tokens.len > max_seq) return error.SequenceTooLong;
            try self.patchRuntimeBindings(tokens, position);
            self.device.execute();
            return self.logits_buf;
        }

        pub fn getProgram(self: *const Self) backend_mod.DeviceProgram {
            return self.device.getProgram();
        }

        pub fn getRuntimeProfile(self: *const Self) ?*profile.RuntimeProfile {
            return self.device.getRuntimeProfile();
        }

        fn patchRuntimeBindings(self: *Self, tokens: []const usize, position: usize) !void {
            const tok_data = self.session.model.token_embed.inner.data;
            for (tokens, 0..) |token_id, i| {
                if (token_id >= config.vocab_size) return error.TokenIdOutOfRange;
                @memcpy(
                    self.plan.token_input.data[i * d_model ..][0..d_model],
                    tok_data[token_id * d_model ..][0..d_model],
                );
            }

            for (0..tokens.len) |j| {
                const col = self.plan.attn_mask.data[j * max_seq ..][0..max_seq];
                const valid_upto = position + j + 1;
                @memset(col[0..valid_upto], 0);
                if (valid_upto < max_seq) @memset(col[valid_upto..], -std.math.inf(T));
            }

            const rope = &self.session.model.blocks[0].rope;
            for (self.plan.trace.layers) |lt| {
                const buf = lt.rope.data;
                for (0..tokens.len) |j| {
                    @memcpy(buf[j * 2 * d_head ..][0..d_head], rope.cos_table.data[(position + j) * d_head ..][0..d_head]);
                    @memcpy(buf[j * 2 * d_head + d_head ..][0..d_head], rope.sin_table.data[(position + j) * d_head ..][0..d_head]);
                }
            }

            for (self.plan.trace.layers) |lt| {
                for (lt.k_write) |node| node.storage_offset = position;
                for (lt.v_write) |node| node.storage_offset = position;
            }
            self.device.patchSliceAssignOffset(@intCast(position));
            self.device.patchAttentionSeqKV(@intCast(position + tokens.len));
        }
    };
}

