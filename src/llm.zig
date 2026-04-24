//! High-level LLM inference facade.
//!
//! This module keeps the public inference surface small while the backend
//! implementation grows toward llama.cpp parity: explicit model/session types,
//! a streaming `prefill`/`step` API, and a weight-storage vocabulary that can
//! represent compressed GGUF weights without forcing the training `Tensor`
//! type to carry inference-only concerns.

const std = @import("std");
const builtin = @import("builtin");

const backend_mod = @import("backend.zig");
const gguf = @import("gguf.zig");
const llama_inference = @import("llama_inference.zig");
const llama_loader = @import("models/llama_loader.zig");
const gguf_loader = @import("models/gguf_loader.zig");
const safetensors = @import("safetensors.zig");
const LlamaConfig = @import("models/llama.zig").LlamaConfig;

pub const stage_plan = @import("llm/stage_plan.zig");
pub const device_prefill = @import("llm/device_prefill.zig");
pub const LlamaDevicePrefill = device_prefill.LlamaDevicePrefill;

/// First-class formats for inference weights. Dense tensors remain the
/// training/autodiff path; quantized variants are for direct inference runtimes.
pub const WeightFormat = enum {
    f32,
    f16,
    q8_0,
    q4_0,
    q4_k,
};

pub const DenseWeightF32 = struct {
    data: []const f32,
    shape: []const usize,
};

pub const DenseWeightF16 = struct {
    data: []const f16,
    shape: []const usize,
};

pub const QuantizedWeight = struct {
    data: []const u8,
    rows: usize,
    cols: usize,
    block_size: usize,
    type_size: usize,
};

pub const WeightStorage = union(WeightFormat) {
    f32: DenseWeightF32,
    f16: DenseWeightF16,
    q8_0: QuantizedWeight,
    q4_0: QuantizedWeight,
    q4_k: QuantizedWeight,

    pub fn format(self: WeightStorage) WeightFormat {
        return std.meta.activeTag(self);
    }
};

pub const BackendPreference = enum {
    auto,
    cpu,
    metal,
    wgpu,

    pub fn resolve(self: BackendPreference) BackendPreference {
        return switch (self) {
            .auto => defaultBackend(),
            else => self,
        };
    }
};

pub const LoadOptions = struct {
    backend: BackendPreference = .auto,
    weight_format: ?WeightFormat = null,
    context_len: ?usize = null,
    prefill_chunk: ?usize = null,
    quantize_weights: bool = false,
    quantize_kv: bool = false,
};

pub const ModelFileKind = enum {
    safetensors,
    gguf,
};

pub const PerformanceTarget = struct {
    prompt_ratio_min: f32 = 0.90,
    decode_ratio_min: f32 = 0.90,
    memory_ratio_max: f32 = 1.15,
};

pub const smollm_parity_target = PerformanceTarget{};

pub fn defaultBackend() BackendPreference {
    return if (builtin.os.tag == .macos) .metal else .cpu;
}

pub fn inferFileKind(path: []const u8) ?ModelFileKind {
    if (endsWithIgnoreCase(path, ".safetensors")) return .safetensors;
    if (endsWithIgnoreCase(path, ".gguf")) return .gguf;
    return null;
}

pub fn weightFormatFromGGML(t: gguf.GGMLType) ?WeightFormat {
    return switch (t) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
        .q4_0 => .q4_0,
        .q4_k => .q4_k,
        else => null,
    };
}

pub fn capabilitiesOf(backend: backend_mod.Backend) backend_mod.Capabilities {
    return backend.capabilities;
}

/// Thin typed wrapper over the existing persistent LLaMA inference session.
/// Users get `prefill(tokens)` and `step(token)` without touching graph internals.
pub fn LlamaSession(comptime T: type, comptime config: LlamaConfig) type {
    const Inner = llama_inference.LlamaInferenceSession(T, config);

    return struct {
        const Self = @This();

        inner: Inner,

        pub fn init(alloc: std.mem.Allocator) !Self {
            return .{ .inner = try Inner.init(alloc) };
        }

        pub fn initWithBackend(alloc: std.mem.Allocator, backend: ?backend_mod.Backend) !Self {
            return .{ .inner = try Inner.initWithBackend(alloc, backend) };
        }

        pub fn deinit(self: *Self) void {
            self.inner.deinit();
        }

        pub fn reset(self: *Self) void {
            self.inner.reset();
        }

        pub fn position(self: *const Self) usize {
            return self.inner.position();
        }

        pub fn prefill(self: *Self, tokens: []const usize) ![]const T {
            return self.inner.prefill(tokens);
        }

        pub fn step(self: *Self, token: usize) ![]const T {
            return self.inner.step(token);
        }

        pub fn loadSafetensors(self: *Self, alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
            var sf = try safetensors.SafetensorsFile.open(alloc, path, io);
            defer sf.deinit();
            self.inner.clearDirectQuantWeights();
            try llama_loader.loadLlama(T, config, &self.inner.model, &sf);
        }

        /// Compatibility path for GGUF correctness and diagnostics. Direct
        /// compressed GGUF execution is represented by `WeightStorage` and is
        /// the next runtime step for parity work.
        pub fn loadGGUFDequantized(self: *Self, alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
            var gf = try gguf.GGUFFile.open(alloc, io, path);
            defer gf.deinit();
            self.inner.clearDirectQuantWeights();
            try gguf_loader.loadDequantized(T, config, &self.inner.model, &gf);
        }

        pub fn loadGGUFDirectQuantized(self: *Self, alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
            var gf = try gguf.GGUFFile.open(alloc, io, path);
            defer gf.deinit();
            try self.inner.loadGGUFDirectQuantized(&gf);
        }

        pub fn load(self: *Self, alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
            return switch (inferFileKind(path) orelse return error.UnknownModelFormat) {
                .safetensors => self.loadSafetensors(alloc, io, path),
                .gguf => self.loadGGUFDirectQuantized(alloc, io, path),
            };
        }
    };
}

fn endsWithIgnoreCase(value: []const u8, suffix: []const u8) bool {
    if (value.len < suffix.len) return false;
    const tail = value[value.len - suffix.len ..];
    for (tail, suffix) |a, b| {
        if (std.ascii.toLower(a) != std.ascii.toLower(b)) return false;
    }
    return true;
}

const testing = std.testing;

test "llm facade infers model file kind" {
    try testing.expectEqual(ModelFileKind.safetensors, inferFileKind("model.safetensors").?);
    try testing.expectEqual(ModelFileKind.safetensors, inferFileKind("MODEL.SAFETENSORS").?);
    try testing.expectEqual(ModelFileKind.gguf, inferFileKind("model.Q8_0.GGUF").?);
    try testing.expectEqual(@as(?ModelFileKind, null), inferFileKind("model.bin"));
}

test "llm facade maps GGML storage formats" {
    try testing.expectEqual(WeightFormat.f16, weightFormatFromGGML(.f16).?);
    try testing.expectEqual(WeightFormat.q8_0, weightFormatFromGGML(.q8_0).?);
    try testing.expectEqual(WeightFormat.q4_k, weightFormatFromGGML(.q4_k).?);
    try testing.expectEqual(@as(?WeightFormat, null), weightFormatFromGGML(.q5_k));
}
