const std = @import("std");
const testing = std.testing;
pub const Tensor = @import("tensor.zig").Tensor;
pub const IndexTensor = @import("index.zig").IndexTensor;
pub const max_dims = @import("tensor.zig").max_dims;
pub const ComputeGraph = @import("graph.zig").ComputeGraph;
pub const Op = @import("op.zig").Op;

pub const shaped = @import("shaped.zig");
pub const Shaped = shaped.Shaped;
pub const ShapedTensor = shaped.ShapedTensor;

pub const models = @import("models.zig");
pub const optim = @import("optim.zig");
pub const loss = @import("loss.zig");
pub const nn = @import("nn.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const comptime_model = @import("comptime_model.zig");
pub const backend = @import("backend.zig");
pub const backend_program = @import("backend/program.zig");
pub const backend_reference = @import("backend/reference.zig");
pub const backend_conformance = @import("backend/conformance.zig");
pub const backend_cpu = @import("backend/cpu.zig");
pub const backend_metal = if (@import("builtin").os.tag == .macos) @import("backend/metal.zig") else struct {};
pub const backend_wgpu = if (@import("zgml_options").use_wgpu) @import("backend/wgpu.zig") else struct {};
pub const llm = @import("llm.zig");
pub const llm_stage_plan = @import("llm/stage_plan.zig");
pub const gguf = @import("gguf.zig");
pub const quant = @import("quant.zig");
pub const safetensors = @import("safetensors.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const inference = @import("inference.zig");
pub const inference_utils = @import("inference_utils.zig");
pub const llama_inference = @import("llama_inference.zig");
pub const device_inference = @import("device_inference.zig");
pub const profile = @import("profile.zig");
pub const data = @import("data.zig");

test "ref all decls" {
    _ = testing.refAllDecls(models);
    _ = testing.refAllDecls(optim);
    _ = testing.refAllDecls(loss);
    _ = @import("nn.zig");
    _ = @import("checkpoint.zig");
    _ = @import("index.zig");
    _ = @import("shaped.zig");
    _ = @import("comptime_model.zig");
    _ = @import("backend.zig");
    _ = @import("backend/program.zig");
    _ = @import("backend/reference.zig");
    _ = @import("backend/conformance.zig");
    _ = @import("backend/cpu.zig");
    if (@import("builtin").os.tag == .macos) {
        _ = @import("backend/metal.zig");
    }
    if (@import("zgml_options").use_wgpu) {
        _ = @import("backend/wgpu.zig");
    }
    _ = @import("llm.zig");
    _ = @import("llm/stage_plan.zig");
    _ = @import("gguf.zig");
    _ = @import("quant.zig");
    _ = @import("safetensors.zig");
    _ = @import("tokenizer.zig");
    _ = @import("inference_utils.zig");
    _ = @import("llama_inference.zig");
    _ = @import("profile.zig");
    _ = @import("data.zig");
}
