//! Opt-in wgpu-native backend prototype.
//!
//! This module is intentionally narrow: it proves a real WebGPU/wgpu compute
//! dispatch can consume a zgml DeviceProgram/ProgramStencil shape without CPU
//! fallback. It is not wired into public backend selection yet.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const DeviceInference = @import("../device_inference.zig").DeviceInference;
const ComputeGraph = @import("../graph.zig").ComputeGraph;
const Tensor = @import("../tensor.zig").Tensor;
const program_mod = @import("program.zig");
const profile_mod = @import("../profile.zig");
const options = @import("zgml_options");

const c = if (options.use_wgpu) @cImport({
    @cInclude("webgpu/webgpu.h");
    @cInclude("webgpu/wgpu.h");
}) else struct {};

const wait_timeout_ns: u64 = 10_000_000_000;
const max_fused_elementwise_steps: usize = 8;
const max_fused_elementwise_secondaries: usize = 6;
const max_matmul_fused_elementwise_secondaries: usize = 5;
const max_matvec_fused_elementwise_secondaries: usize = 5;
const max_qmatvec_fused_elementwise_secondaries: usize = 4;
const max_dispatch_ops: usize = 3;
const max_attention_seq_kv: u32 = 2048;
const max_attention_d_head: u32 = 256;
const storage_buffer_binding_alignment: u64 = 256;
const wgsl_linear =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  k: u32,
    \\  n: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: Values;
    \\@group(0) @binding(1) var<storage, read> weights: Values;
    \\@group(0) @binding(2) var<storage, read> bias: Values;
    \\@group(0) @binding(3) var<storage, read_write> output: Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let j = gid.x;
    \\  if (j >= params.n) {
    \\    return;
    \\  }
    \\  var sum = bias.data[j];
    \\  for (var k: u32 = 0u; k < params.k; k = k + 1u) {
    \\    sum = sum + input.data[k] * weights.data[k * params.n + j];
    \\  }
    \\  output.data[j] = sum;
    \\}
;
const wgsl_matmul =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  m: u32,
    \\  n: u32,
    \\  k: u32,
    \\  a_row_stride: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  dst_row_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> a: Values;
    \\@group(0) @binding(1) var<storage, read> b: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.m * params.n;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx / params.n;
    \\  let col = idx - row * params.n;
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + row * params.a_row_stride + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    sum = sum + a.data[ai] * b.data[bi];
    \\  }
    \\  output.data[params.dst_offset + row * params.dst_row_stride + col] = sum;
    \\}
;
const wgsl_matmul_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  m: u32,
    \\  n: u32,
    \\  k: u32,
    \\  a_row_stride: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  secondary_offset: u32,
    \\  dst_offset: u32,
    \\  is_swapped: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> a: Values;
    \\@group(0) @binding(1) var<storage, read> b: Values;
    \\@group(0) @binding(2) var<storage, read> secondary: Values;
    \\@group(0) @binding(3) var<storage, read_write> output: Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.m * params.n;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx / params.n;
    \\  let col = idx - row * params.n;
    \\  var projected = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + row * params.a_row_stride + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    projected = projected + a.data[ai] * b.data[bi];
    \\  }
    \\  let other = secondary.data[params.secondary_offset + idx];
    \\  var value = projected;
    \\  if (params.op == 1u) {
    \\    value = select(projected + other, other + projected, params.is_swapped != 0u);
    \\  } else if (params.op == 2u) {
    \\    value = select(projected * other, other * projected, params.is_swapped != 0u);
    \\  }
    \\  output.data[params.dst_offset + idx] = value;
    \\}
;
const wgsl_matmul_fused_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  m: u32,
    \\  n: u32,
    \\  k: u32,
    \\  a_row_stride: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  dst_row_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  primary_dst_offset: u32,
    \\  ew_dst_offset: u32,
    \\  n_steps: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\  op0: u32,
    \\  op1: u32,
    \\  op2: u32,
    \\  op3: u32,
    \\  op4: u32,
    \\  op5: u32,
    \\  op6: u32,
    \\  op7: u32,
    \\  is_swapped0: u32,
    \\  is_swapped1: u32,
    \\  is_swapped2: u32,
    \\  is_swapped3: u32,
    \\  is_swapped4: u32,
    \\  is_swapped5: u32,
    \\  is_swapped6: u32,
    \\  is_swapped7: u32,
    \\  secondary_slot0: u32,
    \\  secondary_slot1: u32,
    \\  secondary_slot2: u32,
    \\  secondary_slot3: u32,
    \\  secondary_slot4: u32,
    \\  secondary_slot5: u32,
    \\  secondary_slot6: u32,
    \\  secondary_slot7: u32,
    \\  secondary_offset0: u32,
    \\  secondary_offset1: u32,
    \\  secondary_offset2: u32,
    \\  secondary_offset3: u32,
    \\  secondary_offset4: u32,
    \\  secondary_offset5: u32,
    \\  secondary_offset6: u32,
    \\  secondary_offset7: u32,
    \\  secondary_is_primary0: u32,
    \\  secondary_is_primary1: u32,
    \\  secondary_is_primary2: u32,
    \\  secondary_is_primary3: u32,
    \\  secondary_is_primary4: u32,
    \\  secondary_is_primary5: u32,
    \\  secondary_is_primary6: u32,
    \\  secondary_is_primary7: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> a: Values;
    \\@group(0) @binding(1) var<storage, read> b: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<storage, read> secondary0: Values;
    \\@group(0) @binding(4) var<storage, read> secondary1: Values;
    \\@group(0) @binding(5) var<storage, read> secondary2: Values;
    \\@group(0) @binding(6) var<storage, read> secondary3: Values;
    \\@group(0) @binding(7) var<storage, read> secondary4: Values;
    \\@group(0) @binding(8) var<uniform> params: Params;
    \\
    \\fn gelu(x: f32) -> f32 {
    \\  let c = 0.7978845608f * (x + 0.044715f * x * x * x);
    \\  let e2c = exp(c + c);
    \\  return 0.5f * x * (1.0f + (e2c - 1.0f) / (e2c + 1.0f));
    \\}
    \\
    \\fn sgn(x: f32) -> f32 {
    \\  if (x > 0.0f) {
    \\    return 1.0f;
    \\  }
    \\  if (x < 0.0f) {
    \\    return -1.0f;
    \\  }
    \\  return 0.0f;
    \\}
    \\
    \\fn step_value(x: f32) -> f32 {
    \\  return select(0.0f, 1.0f, x > 0.0f);
    \\}
    \\
    \\fn step_op(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.op0; }
    \\    case 1u: { return params.op1; }
    \\    case 2u: { return params.op2; }
    \\    case 3u: { return params.op3; }
    \\    case 4u: { return params.op4; }
    \\    case 5u: { return params.op5; }
    \\    case 6u: { return params.op6; }
    \\    case 7u: { return params.op7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_is_swapped(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.is_swapped0; }
    \\    case 1u: { return params.is_swapped1; }
    \\    case 2u: { return params.is_swapped2; }
    \\    case 3u: { return params.is_swapped3; }
    \\    case 4u: { return params.is_swapped4; }
    \\    case 5u: { return params.is_swapped5; }
    \\    case 6u: { return params.is_swapped6; }
    \\    case 7u: { return params.is_swapped7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_slot(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_slot0; }
    \\    case 1u: { return params.secondary_slot1; }
    \\    case 2u: { return params.secondary_slot2; }
    \\    case 3u: { return params.secondary_slot3; }
    \\    case 4u: { return params.secondary_slot4; }
    \\    case 5u: { return params.secondary_slot5; }
    \\    case 6u: { return params.secondary_slot6; }
    \\    case 7u: { return params.secondary_slot7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_offset(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_offset0; }
    \\    case 1u: { return params.secondary_offset1; }
    \\    case 2u: { return params.secondary_offset2; }
    \\    case 3u: { return params.secondary_offset3; }
    \\    case 4u: { return params.secondary_offset4; }
    \\    case 5u: { return params.secondary_offset5; }
    \\    case 6u: { return params.secondary_offset6; }
    \\    case 7u: { return params.secondary_offset7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_is_primary(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_is_primary0; }
    \\    case 1u: { return params.secondary_is_primary1; }
    \\    case 2u: { return params.secondary_is_primary2; }
    \\    case 3u: { return params.secondary_is_primary3; }
    \\    case 4u: { return params.secondary_is_primary4; }
    \\    case 5u: { return params.secondary_is_primary5; }
    \\    case 6u: { return params.secondary_is_primary6; }
    \\    case 7u: { return params.secondary_is_primary7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn secondary_value(slot: u32, idx: u32) -> f32 {
    \\  switch (slot) {
    \\    case 0u: { return secondary0.data[idx]; }
    \\    case 1u: { return secondary1.data[idx]; }
    \\    case 2u: { return secondary2.data[idx]; }
    \\    case 3u: { return secondary3.data[idx]; }
    \\    case 4u: { return secondary4.data[idx]; }
    \\    default: { return secondary0.data[idx]; }
    \\  }
    \\}
    \\
    \\fn apply_unary(op: u32, x: f32) -> f32 {
    \\  switch (op) {
    \\    case 3u: { return -x; }
    \\    case 4u: { return abs(x); }
    \\    case 5u: { return sgn(x); }
    \\    case 6u: { return step_value(x); }
    \\    case 7u: { return max(x, 0.0f); }
    \\    case 8u: { return sqrt(x); }
    \\    case 9u: { return 1.0f / x; }
    \\    case 10u: { return exp(x); }
    \\    case 11u: { return log(x); }
    \\    case 12u: { return gelu(x); }
    \\    case 13u: { return x * x; }
    \\    default: { return x; }
    \\  }
    \\}
    \\
    \\fn dot_cell(row: u32, col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + row * params.a_row_stride + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    sum = sum + a.data[ai] * b.data[bi];
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.m * params.n;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx / params.n;
    \\  let col = idx - row * params.n;
    \\  let projected = dot_cell(row, col);
    \\  var v = projected;
    \\  var step = 0u;
    \\  loop {
    \\    if (step >= params.n_steps) {
    \\      break;
    \\    }
    \\    let op = step_op(step);
    \\    if (op == 1u || op == 2u) {
    \\      var other = projected;
    \\      if (step_secondary_is_primary(step) == 0u) {
    \\        other = secondary_value(step_secondary_slot(step), step_secondary_offset(step) + idx);
    \\      }
    \\      if (op == 1u) {
    \\        v = select(v + other, other + v, step_is_swapped(step) != 0u);
    \\      } else {
    \\        v = select(v * other, other * v, step_is_swapped(step) != 0u);
    \\      }
    \\    } else {
    \\      v = apply_unary(op, v);
    \\    }
    \\    step = step + 1u;
    \\  }
    \\  output.data[params.ew_dst_offset + idx] = v;
    \\}
;
const wgsl_matvec_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  n: u32,
    \\  k: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  secondary_offset: u32,
    \\  dst_offset: u32,
    \\  is_swapped: u32,
    \\  _pad0: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: Values;
    \\@group(0) @binding(1) var<storage, read> weights: Values;
    \\@group(0) @binding(2) var<storage, read> secondary: Values;
    \\@group(0) @binding(3) var<storage, read_write> output: Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let col = gid.x;
    \\  if (col >= params.n) {
    \\    return;
    \\  }
    \\  var projected = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    projected = projected + input.data[ai] * weights.data[bi];
    \\  }
    \\  let other = secondary.data[params.secondary_offset + col];
    \\  var value = projected;
    \\  if (params.op == 1u) {
    \\    value = select(projected + other, other + projected, params.is_swapped != 0u);
    \\  } else if (params.op == 2u) {
    \\    value = select(projected * other, other * projected, params.is_swapped != 0u);
    \\  }
    \\  output.data[params.dst_offset + col] = value;
    \\}
;
const wgsl_matvec_fused_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  dst_offset: u32,
    \\  n_steps: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\  op0: u32,
    \\  op1: u32,
    \\  op2: u32,
    \\  op3: u32,
    \\  op4: u32,
    \\  op5: u32,
    \\  op6: u32,
    \\  op7: u32,
    \\  is_swapped0: u32,
    \\  is_swapped1: u32,
    \\  is_swapped2: u32,
    \\  is_swapped3: u32,
    \\  is_swapped4: u32,
    \\  is_swapped5: u32,
    \\  is_swapped6: u32,
    \\  is_swapped7: u32,
    \\  secondary_slot0: u32,
    \\  secondary_slot1: u32,
    \\  secondary_slot2: u32,
    \\  secondary_slot3: u32,
    \\  secondary_slot4: u32,
    \\  secondary_slot5: u32,
    \\  secondary_slot6: u32,
    \\  secondary_slot7: u32,
    \\  secondary_offset0: u32,
    \\  secondary_offset1: u32,
    \\  secondary_offset2: u32,
    \\  secondary_offset3: u32,
    \\  secondary_offset4: u32,
    \\  secondary_offset5: u32,
    \\  secondary_offset6: u32,
    \\  secondary_offset7: u32,
    \\  secondary_is_primary0: u32,
    \\  secondary_is_primary1: u32,
    \\  secondary_is_primary2: u32,
    \\  secondary_is_primary3: u32,
    \\  secondary_is_primary4: u32,
    \\  secondary_is_primary5: u32,
    \\  secondary_is_primary6: u32,
    \\  secondary_is_primary7: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: Values;
    \\@group(0) @binding(1) var<storage, read> weights: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<storage, read> secondary0: Values;
    \\@group(0) @binding(4) var<storage, read> secondary1: Values;
    \\@group(0) @binding(5) var<storage, read> secondary2: Values;
    \\@group(0) @binding(6) var<storage, read> secondary3: Values;
    \\@group(0) @binding(7) var<storage, read> secondary4: Values;
    \\@group(0) @binding(8) var<uniform> params: Params;
    \\
    \\fn gelu(x: f32) -> f32 {
    \\  let c = 0.7978845608f * (x + 0.044715f * x * x * x);
    \\  let e2c = exp(c + c);
    \\  return 0.5f * x * (1.0f + (e2c - 1.0f) / (e2c + 1.0f));
    \\}
    \\
    \\fn sgn(x: f32) -> f32 {
    \\  if (x > 0.0f) {
    \\    return 1.0f;
    \\  }
    \\  if (x < 0.0f) {
    \\    return -1.0f;
    \\  }
    \\  return 0.0f;
    \\}
    \\
    \\fn step_value(x: f32) -> f32 {
    \\  return select(0.0f, 1.0f, x > 0.0f);
    \\}
    \\
    \\fn step_op(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.op0; }
    \\    case 1u: { return params.op1; }
    \\    case 2u: { return params.op2; }
    \\    case 3u: { return params.op3; }
    \\    case 4u: { return params.op4; }
    \\    case 5u: { return params.op5; }
    \\    case 6u: { return params.op6; }
    \\    case 7u: { return params.op7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_is_swapped(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.is_swapped0; }
    \\    case 1u: { return params.is_swapped1; }
    \\    case 2u: { return params.is_swapped2; }
    \\    case 3u: { return params.is_swapped3; }
    \\    case 4u: { return params.is_swapped4; }
    \\    case 5u: { return params.is_swapped5; }
    \\    case 6u: { return params.is_swapped6; }
    \\    case 7u: { return params.is_swapped7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_slot(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_slot0; }
    \\    case 1u: { return params.secondary_slot1; }
    \\    case 2u: { return params.secondary_slot2; }
    \\    case 3u: { return params.secondary_slot3; }
    \\    case 4u: { return params.secondary_slot4; }
    \\    case 5u: { return params.secondary_slot5; }
    \\    case 6u: { return params.secondary_slot6; }
    \\    case 7u: { return params.secondary_slot7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_offset(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_offset0; }
    \\    case 1u: { return params.secondary_offset1; }
    \\    case 2u: { return params.secondary_offset2; }
    \\    case 3u: { return params.secondary_offset3; }
    \\    case 4u: { return params.secondary_offset4; }
    \\    case 5u: { return params.secondary_offset5; }
    \\    case 6u: { return params.secondary_offset6; }
    \\    case 7u: { return params.secondary_offset7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_is_primary(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_is_primary0; }
    \\    case 1u: { return params.secondary_is_primary1; }
    \\    case 2u: { return params.secondary_is_primary2; }
    \\    case 3u: { return params.secondary_is_primary3; }
    \\    case 4u: { return params.secondary_is_primary4; }
    \\    case 5u: { return params.secondary_is_primary5; }
    \\    case 6u: { return params.secondary_is_primary6; }
    \\    case 7u: { return params.secondary_is_primary7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn secondary_value(slot: u32, idx: u32) -> f32 {
    \\  switch (slot) {
    \\    case 0u: { return secondary0.data[idx]; }
    \\    case 1u: { return secondary1.data[idx]; }
    \\    case 2u: { return secondary2.data[idx]; }
    \\    case 3u: { return secondary3.data[idx]; }
    \\    case 4u: { return secondary4.data[idx]; }
    \\    default: { return secondary0.data[idx]; }
    \\  }
    \\}
    \\
    \\fn apply_unary(op: u32, x: f32) -> f32 {
    \\  switch (op) {
    \\    case 3u: { return -x; }
    \\    case 4u: { return abs(x); }
    \\    case 5u: { return sgn(x); }
    \\    case 6u: { return step_value(x); }
    \\    case 7u: { return max(x, 0.0f); }
    \\    case 8u: { return sqrt(x); }
    \\    case 9u: { return 1.0f / x; }
    \\    case 10u: { return exp(x); }
    \\    case 11u: { return log(x); }
    \\    case 12u: { return gelu(x); }
    \\    case 13u: { return x * x; }
    \\    default: { return x; }
    \\  }
    \\}
    \\
    \\fn dot_col(col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    sum = sum + input.data[ai] * weights.data[bi];
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let col = gid.x;
    \\  if (col >= params.n) {
    \\    return;
    \\  }
    \\  let projected = dot_col(col);
    \\  var v = projected;
    \\  var step = 0u;
    \\  loop {
    \\    if (step >= params.n_steps) {
    \\      break;
    \\    }
    \\    let op = step_op(step);
    \\    if (op == 1u || op == 2u) {
    \\      var other = projected;
    \\      if (step_secondary_is_primary(step) == 0u) {
    \\        other = secondary_value(step_secondary_slot(step), step_secondary_offset(step) + col);
    \\      }
    \\      if (op == 1u) {
    \\        v = select(v + other, other + v, step_is_swapped(step) != 0u);
    \\      } else {
    \\        v = select(v * other, other * v, step_is_swapped(step) != 0u);
    \\      }
    \\    } else {
    \\      v = apply_unary(op, v);
    \\    }
    \\    step = step + 1u;
    \\  }
    \\  output.data[params.dst_offset + col] = v;
    \\}
;
const wgsl_matvec_slice =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  cells: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  slice_src_col_start: u32,
    \\  slice_dst_offset: u32,
    \\  slice_dst_row_stride: u32,
    \\  slice_dst_col_stride: u32,
    \\  slice_rows: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: Values;
    \\@group(0) @binding(1) var<storage, read> weights: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<uniform> params: Params;
    \\
    \\fn dot_col(col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    sum = sum + input.data[ai] * weights.data[bi];
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let local = gid.x;
    \\  if (local >= params.cells) {
    \\    return;
    \\  }
    \\  let src_col = params.slice_src_col_start + local;
    \\  let value = dot_col(src_col);
    \\  let row = local % params.slice_rows;
    \\  let col = local / params.slice_rows;
    \\  output.data[params.slice_dst_offset + row * params.slice_dst_row_stride + col * params.slice_dst_col_stride] = value;
    \\}
;
const wgsl_matvec_rope_slice =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  half_d: u32,
    \\  a_col_stride: u32,
    \\  b_row_stride: u32,
    \\  b_col_stride: u32,
    \\  a_offset: u32,
    \\  b_offset: u32,
    \\  rope_src_col_start: u32,
    \\  cs_off: u32,
    \\  cs_cs: u32,
    \\  rope_dst_offset: u32,
    \\  rope_dst_row_stride: u32,
    \\  rope_dst_col_stride: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: Values;
    \\@group(0) @binding(1) var<storage, read> weights: Values;
    \\@group(0) @binding(2) var<storage, read> cos_sin: Values;
    \\@group(0) @binding(3) var<storage, read_write> output: Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\fn dot_col(col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let ai = params.a_offset + kk * params.a_col_stride;
    \\    let bi = params.b_offset + kk * params.b_row_stride + col * params.b_col_stride;
    \\    sum = sum + input.data[ai] * weights.data[bi];
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let local = gid.x;
    \\  if (local >= params.half_d) {
    \\    return;
    \\  }
    \\  let lo_col = params.rope_src_col_start + local;
    \\  let hi_col = lo_col + params.half_d;
    \\  let lo = dot_col(lo_col);
    \\  let hi = dot_col(hi_col);
    \\  let cos_v = cos_sin.data[params.cs_off + local];
    \\  let sin_v = cos_sin.data[params.cs_off + params.half_d * 2u + local];
    \\  let y_lo = lo * cos_v - hi * sin_v;
    \\  let y_hi = hi * cos_v + lo * sin_v;
    \\  output.data[params.rope_dst_offset + local * params.rope_dst_row_stride] = y_lo;
    \\  output.data[params.rope_dst_offset + (local + params.half_d) * params.rope_dst_row_stride] = y_hi;
    \\}
;
const wgsl_qmatmul =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  m: u32,
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  input_offset: u32,
    \\  input_row_stride: u32,
    \\  dst_offset: u32,
    \\  dst_row_stride: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.m * params.n;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx / params.n;
    \\  let col = idx - row * params.n;
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    let input_idx = params.input_offset + row * params.input_row_stride + kk;
    \\    sum = sum + input.data[input_idx] * q * scale;
    \\  }
    \\  output.data[params.dst_offset + row * params.dst_row_stride + col] = sum;
    \\}
;
const wgsl_qmatmul_elementwise =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  m: u32,
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  input_offset: u32,
    \\  input_row_stride: u32,
    \\  secondary_offset: u32,
    \\  dst_offset: u32,
    \\  is_swapped: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read> secondary: F32Values;
    \\@group(0) @binding(4) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.m * params.n;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx / params.n;
    \\  let col = idx - row * params.n;
    \\  var projected = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    let input_idx = params.input_offset + row * params.input_row_stride + kk;
    \\    projected = projected + input.data[input_idx] * q * scale;
    \\  }
    \\  let other = secondary.data[params.secondary_offset + idx];
    \\  var value = projected;
    \\  if (params.op == 1u) {
    \\    value = select(projected + other, other + projected, params.is_swapped != 0u);
    \\  } else if (params.op == 2u) {
    \\    value = select(projected * other, other * projected, params.is_swapped != 0u);
    \\  }
    \\  output.data[params.dst_offset + idx] = value;
    \\}
;
const wgsl_qmatvec_slice =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  cells: u32,
    \\  input_offset: u32,
    \\  slice_src_col_start: u32,
    \\  slice_dst_offset: u32,
    \\  slice_dst_row_stride: u32,
    \\  slice_dst_col_stride: u32,
    \\  slice_rows: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(4) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let local = gid.x;
    \\  if (local >= params.cells) {
    \\    return;
    \\  }
    \\  let q_col = params.slice_src_col_start + local;
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + q_col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    sum = sum + input.data[params.input_offset + kk] * q * scale;
    \\  }
    \\  let row = local % params.slice_rows;
    \\  let col = local / params.slice_rows;
    \\  output.data[params.slice_dst_offset + row * params.slice_dst_row_stride + col * params.slice_dst_col_stride] = sum;
    \\}
;
const wgsl_qmatvec_rope_slice =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  half_d: u32,
    \\  input_offset: u32,
    \\  rope_src_col_start: u32,
    \\  cs_off: u32,
    \\  cs_cs: u32,
    \\  rope_dst_offset: u32,
    \\  rope_dst_row_stride: u32,
    \\  rope_dst_col_stride: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read> cos_sin: F32Values;
    \\@group(0) @binding(4) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\fn dot_col(col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    sum = sum + input.data[params.input_offset + kk] * q * scale;
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let local = gid.x;
    \\  if (local >= params.half_d) {
    \\    return;
    \\  }
    \\  let lo_col = params.rope_src_col_start + local;
    \\  let hi_col = lo_col + params.half_d;
    \\  let lo = dot_col(lo_col);
    \\  let hi = dot_col(hi_col);
    \\  let cos_v = cos_sin.data[params.cs_off + local];
    \\  let sin_v = cos_sin.data[params.cs_off + params.half_d * 2u + local];
    \\  let y_lo = lo * cos_v - hi * sin_v;
    \\  let y_hi = hi * cos_v + lo * sin_v;
    \\  output.data[params.rope_dst_offset + local * params.rope_dst_row_stride] = y_lo;
    \\  output.data[params.rope_dst_offset + (local + params.half_d) * params.rope_dst_row_stride] = y_hi;
    \\}
;
const wgsl_qmatvec_elementwise =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  input_offset: u32,
    \\  secondary_offset: u32,
    \\  dst_offset: u32,
    \\  is_swapped: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read> secondary: F32Values;
    \\@group(0) @binding(4) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let col = gid.x;
    \\  if (col >= params.n) {
    \\    return;
    \\  }
    \\  var projected = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    projected = projected + input.data[params.input_offset + kk] * q * scale;
    \\  }
    \\  let other = secondary.data[params.secondary_offset + col];
    \\  var value = projected;
    \\  if (params.op == 1u) {
    \\    value = select(projected + other, other + projected, params.is_swapped != 0u);
    \\  } else if (params.op == 2u) {
    \\    value = select(projected * other, other * projected, params.is_swapped != 0u);
    \\  }
    \\  output.data[params.dst_offset + col] = value;
    \\}
;
const wgsl_qmatvec_fused_elementwise =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  input_offset: u32,
    \\  dst_offset: u32,
    \\  n_steps: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  op0: u32,
    \\  op1: u32,
    \\  op2: u32,
    \\  op3: u32,
    \\  op4: u32,
    \\  op5: u32,
    \\  op6: u32,
    \\  op7: u32,
    \\  is_swapped0: u32,
    \\  is_swapped1: u32,
    \\  is_swapped2: u32,
    \\  is_swapped3: u32,
    \\  is_swapped4: u32,
    \\  is_swapped5: u32,
    \\  is_swapped6: u32,
    \\  is_swapped7: u32,
    \\  secondary_slot0: u32,
    \\  secondary_slot1: u32,
    \\  secondary_slot2: u32,
    \\  secondary_slot3: u32,
    \\  secondary_slot4: u32,
    \\  secondary_slot5: u32,
    \\  secondary_slot6: u32,
    \\  secondary_slot7: u32,
    \\  secondary_offset0: u32,
    \\  secondary_offset1: u32,
    \\  secondary_offset2: u32,
    \\  secondary_offset3: u32,
    \\  secondary_offset4: u32,
    \\  secondary_offset5: u32,
    \\  secondary_offset6: u32,
    \\  secondary_offset7: u32,
    \\  secondary_is_primary0: u32,
    \\  secondary_is_primary1: u32,
    \\  secondary_is_primary2: u32,
    \\  secondary_is_primary3: u32,
    \\  secondary_is_primary4: u32,
    \\  secondary_is_primary5: u32,
    \\  secondary_is_primary6: u32,
    \\  secondary_is_primary7: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(4) var<storage, read> secondary0: F32Values;
    \\@group(0) @binding(5) var<storage, read> secondary1: F32Values;
    \\@group(0) @binding(6) var<storage, read> secondary2: F32Values;
    \\@group(0) @binding(7) var<storage, read> secondary3: F32Values;
    \\@group(0) @binding(8) var<uniform> params: Params;
    \\
    \\fn gelu(x: f32) -> f32 {
    \\  let c = 0.7978845608f * (x + 0.044715f * x * x * x);
    \\  let e2c = exp(c + c);
    \\  return 0.5f * x * (1.0f + (e2c - 1.0f) / (e2c + 1.0f));
    \\}
    \\
    \\fn sgn(x: f32) -> f32 {
    \\  if (x > 0.0f) {
    \\    return 1.0f;
    \\  }
    \\  if (x < 0.0f) {
    \\    return -1.0f;
    \\  }
    \\  return 0.0f;
    \\}
    \\
    \\fn step_value(x: f32) -> f32 {
    \\  return select(0.0f, 1.0f, x > 0.0f);
    \\}
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\fn step_op(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.op0; }
    \\    case 1u: { return params.op1; }
    \\    case 2u: { return params.op2; }
    \\    case 3u: { return params.op3; }
    \\    case 4u: { return params.op4; }
    \\    case 5u: { return params.op5; }
    \\    case 6u: { return params.op6; }
    \\    case 7u: { return params.op7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_is_swapped(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.is_swapped0; }
    \\    case 1u: { return params.is_swapped1; }
    \\    case 2u: { return params.is_swapped2; }
    \\    case 3u: { return params.is_swapped3; }
    \\    case 4u: { return params.is_swapped4; }
    \\    case 5u: { return params.is_swapped5; }
    \\    case 6u: { return params.is_swapped6; }
    \\    case 7u: { return params.is_swapped7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_slot(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_slot0; }
    \\    case 1u: { return params.secondary_slot1; }
    \\    case 2u: { return params.secondary_slot2; }
    \\    case 3u: { return params.secondary_slot3; }
    \\    case 4u: { return params.secondary_slot4; }
    \\    case 5u: { return params.secondary_slot5; }
    \\    case 6u: { return params.secondary_slot6; }
    \\    case 7u: { return params.secondary_slot7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_offset(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_offset0; }
    \\    case 1u: { return params.secondary_offset1; }
    \\    case 2u: { return params.secondary_offset2; }
    \\    case 3u: { return params.secondary_offset3; }
    \\    case 4u: { return params.secondary_offset4; }
    \\    case 5u: { return params.secondary_offset5; }
    \\    case 6u: { return params.secondary_offset6; }
    \\    case 7u: { return params.secondary_offset7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_is_primary(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_is_primary0; }
    \\    case 1u: { return params.secondary_is_primary1; }
    \\    case 2u: { return params.secondary_is_primary2; }
    \\    case 3u: { return params.secondary_is_primary3; }
    \\    case 4u: { return params.secondary_is_primary4; }
    \\    case 5u: { return params.secondary_is_primary5; }
    \\    case 6u: { return params.secondary_is_primary6; }
    \\    case 7u: { return params.secondary_is_primary7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn secondary_value(slot: u32, idx: u32) -> f32 {
    \\  switch (slot) {
    \\    case 0u: { return secondary0.data[idx]; }
    \\    case 1u: { return secondary1.data[idx]; }
    \\    case 2u: { return secondary2.data[idx]; }
    \\    case 3u: { return secondary3.data[idx]; }
    \\    default: { return secondary0.data[idx]; }
    \\  }
    \\}
    \\
    \\fn apply_unary(op: u32, x: f32) -> f32 {
    \\  switch (op) {
    \\    case 3u: { return -x; }
    \\    case 4u: { return abs(x); }
    \\    case 5u: { return sgn(x); }
    \\    case 6u: { return step_value(x); }
    \\    case 7u: { return max(x, 0.0f); }
    \\    case 8u: { return sqrt(x); }
    \\    case 9u: { return 1.0f / x; }
    \\    case 10u: { return exp(x); }
    \\    case 11u: { return log(x); }
    \\    case 12u: { return gelu(x); }
    \\    case 13u: { return x * x; }
    \\    default: { return x; }
    \\  }
    \\}
    \\
    \\fn dot_col(col: u32) -> f32 {
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    sum = sum + input.data[params.input_offset + kk] * q * scale;
    \\  }
    \\  return sum;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let col = gid.x;
    \\  if (col >= params.n) {
    \\    return;
    \\  }
    \\  let projected = dot_col(col);
    \\  var v = projected;
    \\  var step = 0u;
    \\  loop {
    \\    if (step >= params.n_steps) {
    \\      break;
    \\    }
    \\    let op = step_op(step);
    \\    if (op == 1u || op == 2u) {
    \\      var other = projected;
    \\      if (step_secondary_is_primary(step) == 0u) {
    \\        other = secondary_value(step_secondary_slot(step), step_secondary_offset(step) + col);
    \\      }
    \\      if (op == 1u) {
    \\        v = select(v + other, other + v, step_is_swapped(step) != 0u);
    \\      } else {
    \\        v = select(v * other, other * v, step_is_swapped(step) != 0u);
    \\      }
    \\    } else {
    \\      v = apply_unary(op, v);
    \\    }
    \\    step = step + 1u;
    \\  }
    \\  output.data[params.dst_offset + col] = v;
    \\}
;
const wgsl_qlinear =
    \\struct F32Values {
    \\  data: array<f32>,
    \\};
    \\struct U32Values {
    \\  data: array<u32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  k: u32,
    \\  block_size: u32,
    \\  input_offset: u32,
    \\  bias_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> input: F32Values;
    \\@group(0) @binding(1) var<storage, read> qdata: U32Values;
    \\@group(0) @binding(2) var<storage, read> scales: F32Values;
    \\@group(0) @binding(3) var<storage, read> bias: F32Values;
    \\@group(0) @binding(4) var<storage, read_write> output: F32Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\
    \\fn read_i8(index: u32) -> f32 {
    \\  let word = qdata.data[index / 4u];
    \\  let shift = (index & 3u) * 8u;
    \\  let byte = (word >> shift) & 255u;
    \\  if (byte >= 128u) {
    \\    return f32(i32(byte) - 256);
    \\  }
    \\  return f32(i32(byte));
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let col = gid.x;
    \\  if (col >= params.n) {
    \\    return;
    \\  }
    \\  var sum = 0.0f;
    \\  for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
    \\    let flat = kk * params.n + col;
    \\    let q = read_i8(flat);
    \\    let scale = scales.data[flat / params.block_size];
    \\    sum = sum + input.data[params.input_offset + kk] * q * scale;
    \\  }
    \\  output.data[params.dst_offset + col] = sum + bias.data[params.bias_offset + col];
    \\}
;
const wgsl_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  n: u32,
    \\  dst_offset: u32,
    \\  src0_offset: u32,
    \\  src1_offset: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src0: Values;
    \\@group(0) @binding(1) var<storage, read> src1: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<uniform> params: Params;
    \\
    \\fn gelu(x: f32) -> f32 {
    \\  let c = 0.7978845608f * (x + 0.044715f * x * x * x);
    \\  let e2c = exp(c + c);
    \\  return 0.5f * x * (1.0f + (e2c - 1.0f) / (e2c + 1.0f));
    \\}
    \\
    \\fn sgn(x: f32) -> f32 {
    \\  if (x > 0.0f) {
    \\    return 1.0f;
    \\  }
    \\  if (x < 0.0f) {
    \\    return -1.0f;
    \\  }
    \\  return 0.0f;
    \\}
    \\
    \\fn step_value(x: f32) -> f32 {
    \\  return select(0.0f, 1.0f, x > 0.0f);
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let i = gid.x;
    \\  if (i >= params.n) {
    \\    return;
    \\  }
    \\  let x = src0.data[params.src0_offset + i];
    \\  let y = src1.data[params.src1_offset + i];
    \\  var v = x;
    \\  switch (params.op) {
    \\    case 1u: {
    \\      v = x + y;
    \\    }
    \\    case 2u: {
    \\      v = x * y;
    \\    }
    \\    case 3u: {
    \\      v = -x;
    \\    }
    \\    case 4u: {
    \\      v = abs(x);
    \\    }
    \\    case 5u: {
    \\      v = sgn(x);
    \\    }
    \\    case 6u: {
    \\      v = step_value(x);
    \\    }
    \\    case 7u: {
    \\      v = max(x, 0.0f);
    \\    }
    \\    case 8u: {
    \\      v = sqrt(x);
    \\    }
    \\    case 9u: {
    \\      v = 1.0f / x;
    \\    }
    \\    case 10u: {
    \\      v = exp(x);
    \\    }
    \\    case 11u: {
    \\      v = log(x);
    \\    }
    \\    case 12u: {
    \\      v = gelu(x);
    \\    }
    \\    default: {
    \\      v = x;
    \\    }
    \\  }
    \\  output.data[params.dst_offset + i] = v;
    \\}
;
const wgsl_fused_elementwise =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  n_steps: u32,
    \\  dst_offset: u32,
    \\  src_offset: u32,
    \\  op0: u32,
    \\  op1: u32,
    \\  op2: u32,
    \\  op3: u32,
    \\  op4: u32,
    \\  op5: u32,
    \\  op6: u32,
    \\  op7: u32,
    \\  is_swapped0: u32,
    \\  is_swapped1: u32,
    \\  is_swapped2: u32,
    \\  is_swapped3: u32,
    \\  is_swapped4: u32,
    \\  is_swapped5: u32,
    \\  is_swapped6: u32,
    \\  is_swapped7: u32,
    \\  secondary_slot0: u32,
    \\  secondary_slot1: u32,
    \\  secondary_slot2: u32,
    \\  secondary_slot3: u32,
    \\  secondary_slot4: u32,
    \\  secondary_slot5: u32,
    \\  secondary_slot6: u32,
    \\  secondary_slot7: u32,
    \\  secondary_offset0: u32,
    \\  secondary_offset1: u32,
    \\  secondary_offset2: u32,
    \\  secondary_offset3: u32,
    \\  secondary_offset4: u32,
    \\  secondary_offset5: u32,
    \\  secondary_offset6: u32,
    \\  secondary_offset7: u32,
    \\  secondary_is_repeat0: u32,
    \\  secondary_is_repeat1: u32,
    \\  secondary_is_repeat2: u32,
    \\  secondary_is_repeat3: u32,
    \\  secondary_is_repeat4: u32,
    \\  secondary_is_repeat5: u32,
    \\  secondary_is_repeat6: u32,
    \\  secondary_is_repeat7: u32,
    \\  secondary_repeat_dst_offset0: u32,
    \\  secondary_repeat_dst_offset1: u32,
    \\  secondary_repeat_dst_offset2: u32,
    \\  secondary_repeat_dst_offset3: u32,
    \\  secondary_repeat_dst_offset4: u32,
    \\  secondary_repeat_dst_offset5: u32,
    \\  secondary_repeat_dst_offset6: u32,
    \\  secondary_repeat_dst_offset7: u32,
    \\  repeat_src_offset: u32,
    \\  repeat_src_ne0: u32,
    \\  repeat_src_ne1: u32,
    \\  repeat_src_ne2: u32,
    \\  repeat_src_ne3: u32,
    \\  repeat_src_stride0: u32,
    \\  repeat_src_stride1: u32,
    \\  repeat_src_stride2: u32,
    \\  repeat_src_stride3: u32,
    \\  repeat_dst_stride0: u32,
    \\  repeat_dst_stride1: u32,
    \\  repeat_dst_stride2: u32,
    \\  repeat_dst_stride3: u32,
    \\  _pad_repeat0: u32,
    \\  _pad_repeat1: u32,
    \\  _pad_repeat2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<storage, read> secondary0: Values;
    \\@group(0) @binding(3) var<storage, read> secondary1: Values;
    \\@group(0) @binding(4) var<storage, read> secondary2: Values;
    \\@group(0) @binding(5) var<storage, read> secondary3: Values;
    \\@group(0) @binding(6) var<storage, read> secondary4: Values;
    \\@group(0) @binding(7) var<storage, read> secondary5: Values;
    \\@group(0) @binding(8) var<uniform> params: Params;
    \\
    \\fn gelu(x: f32) -> f32 {
    \\  let c = 0.7978845608f * (x + 0.044715f * x * x * x);
    \\  let e2c = exp(c + c);
    \\  return 0.5f * x * (1.0f + (e2c - 1.0f) / (e2c + 1.0f));
    \\}
    \\
    \\fn sgn(x: f32) -> f32 {
    \\  if (x > 0.0f) {
    \\    return 1.0f;
    \\  }
    \\  if (x < 0.0f) {
    \\    return -1.0f;
    \\  }
    \\  return 0.0f;
    \\}
    \\
    \\fn step_value(x: f32) -> f32 {
    \\  return select(0.0f, 1.0f, x > 0.0f);
    \\}
    \\
    \\fn step_op(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.op0; }
    \\    case 1u: { return params.op1; }
    \\    case 2u: { return params.op2; }
    \\    case 3u: { return params.op3; }
    \\    case 4u: { return params.op4; }
    \\    case 5u: { return params.op5; }
    \\    case 6u: { return params.op6; }
    \\    case 7u: { return params.op7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_is_swapped(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.is_swapped0; }
    \\    case 1u: { return params.is_swapped1; }
    \\    case 2u: { return params.is_swapped2; }
    \\    case 3u: { return params.is_swapped3; }
    \\    case 4u: { return params.is_swapped4; }
    \\    case 5u: { return params.is_swapped5; }
    \\    case 6u: { return params.is_swapped6; }
    \\    case 7u: { return params.is_swapped7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_slot(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_slot0; }
    \\    case 1u: { return params.secondary_slot1; }
    \\    case 2u: { return params.secondary_slot2; }
    \\    case 3u: { return params.secondary_slot3; }
    \\    case 4u: { return params.secondary_slot4; }
    \\    case 5u: { return params.secondary_slot5; }
    \\    case 6u: { return params.secondary_slot6; }
    \\    case 7u: { return params.secondary_slot7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_offset(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_offset0; }
    \\    case 1u: { return params.secondary_offset1; }
    \\    case 2u: { return params.secondary_offset2; }
    \\    case 3u: { return params.secondary_offset3; }
    \\    case 4u: { return params.secondary_offset4; }
    \\    case 5u: { return params.secondary_offset5; }
    \\    case 6u: { return params.secondary_offset6; }
    \\    case 7u: { return params.secondary_offset7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_is_repeat(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_is_repeat0; }
    \\    case 1u: { return params.secondary_is_repeat1; }
    \\    case 2u: { return params.secondary_is_repeat2; }
    \\    case 3u: { return params.secondary_is_repeat3; }
    \\    case 4u: { return params.secondary_is_repeat4; }
    \\    case 5u: { return params.secondary_is_repeat5; }
    \\    case 6u: { return params.secondary_is_repeat6; }
    \\    case 7u: { return params.secondary_is_repeat7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn step_secondary_repeat_dst_offset(step: u32) -> u32 {
    \\  switch (step) {
    \\    case 0u: { return params.secondary_repeat_dst_offset0; }
    \\    case 1u: { return params.secondary_repeat_dst_offset1; }
    \\    case 2u: { return params.secondary_repeat_dst_offset2; }
    \\    case 3u: { return params.secondary_repeat_dst_offset3; }
    \\    case 4u: { return params.secondary_repeat_dst_offset4; }
    \\    case 5u: { return params.secondary_repeat_dst_offset5; }
    \\    case 6u: { return params.secondary_repeat_dst_offset6; }
    \\    case 7u: { return params.secondary_repeat_dst_offset7; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn repeat_src_ne(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.repeat_src_ne0; }
    \\    case 1u: { return params.repeat_src_ne1; }
    \\    case 2u: { return params.repeat_src_ne2; }
    \\    case 3u: { return params.repeat_src_ne3; }
    \\    default: { return 1u; }
    \\  }
    \\}
    \\
    \\fn repeat_src_stride(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.repeat_src_stride0; }
    \\    case 1u: { return params.repeat_src_stride1; }
    \\    case 2u: { return params.repeat_src_stride2; }
    \\    case 3u: { return params.repeat_src_stride3; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn repeat_dst_stride(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.repeat_dst_stride0; }
    \\    case 1u: { return params.repeat_dst_stride1; }
    \\    case 2u: { return params.repeat_dst_stride2; }
    \\    case 3u: { return params.repeat_dst_stride3; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn repeat_secondary_index(step: u32, gid: u32) -> u32 {
    \\  var idx = step_secondary_repeat_dst_offset(step) + gid;
    \\  var src_idx = params.repeat_src_offset;
    \\  var dim: i32 = 3;
    \\  loop {
    \\    if (dim < 0) {
    \\      break;
    \\    }
    \\    let ud = u32(dim);
    \\    let stride = repeat_dst_stride(ud);
    \\    var coord = 0u;
    \\    if (stride != 0u) {
    \\      coord = idx / stride;
    \\      idx = idx % stride;
    \\    }
    \\    let extent = repeat_src_ne(ud);
    \\    if (extent != 0u) {
    \\      src_idx = src_idx + (coord % extent) * repeat_src_stride(ud);
    \\    }
    \\    dim = dim - 1;
    \\  }
    \\  return src_idx;
    \\}
    \\
    \\fn secondary_value(slot: u32, idx: u32) -> f32 {
    \\  switch (slot) {
    \\    case 0u: { return secondary0.data[idx]; }
    \\    case 1u: { return secondary1.data[idx]; }
    \\    case 2u: { return secondary2.data[idx]; }
    \\    case 3u: { return secondary3.data[idx]; }
    \\    case 4u: { return secondary4.data[idx]; }
    \\    case 5u: { return secondary5.data[idx]; }
    \\    default: { return secondary0.data[idx]; }
    \\  }
    \\}
    \\
    \\fn apply_unary(op: u32, x: f32) -> f32 {
    \\  switch (op) {
    \\    case 3u: { return -x; }
    \\    case 4u: { return abs(x); }
    \\    case 5u: { return sgn(x); }
    \\    case 6u: { return step_value(x); }
    \\    case 7u: { return max(x, 0.0f); }
    \\    case 8u: { return sqrt(x); }
    \\    case 9u: { return 1.0f / x; }
    \\    case 10u: { return exp(x); }
    \\    case 11u: { return log(x); }
    \\    case 12u: { return gelu(x); }
    \\    case 13u: { return x * x; }
    \\    default: { return x; }
    \\  }
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let i = gid.x;
    \\  if (i >= params.n) {
    \\    return;
    \\  }
    \\  var v = src.data[params.src_offset + i];
    \\  var step = 0u;
    \\  loop {
    \\    if (step >= params.n_steps) {
    \\      break;
    \\    }
    \\    let op = step_op(step);
    \\    if (op == 1u || op == 2u) {
    \\      let secondary_idx = select(step_secondary_offset(step) + i, repeat_secondary_index(step, i), step_secondary_is_repeat(step) != 0u);
    \\      let other = secondary_value(step_secondary_slot(step), secondary_idx);
    \\      if (op == 1u) {
    \\        v = select(v + other, other + v, step_is_swapped(step) != 0u);
    \\      } else {
    \\        v = select(v * other, other * v, step_is_swapped(step) != 0u);
    \\      }
    \\    } else {
    \\      v = apply_unary(op, v);
    \\    }
    \\    step = step + 1u;
    \\  }
    \\  output.data[params.dst_offset + i] = v;
    \\}
;
const wgsl_repeat =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  n: u32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\  src_ne0: u32,
    \\  src_ne1: u32,
    \\  src_ne2: u32,
    \\  src_ne3: u32,
    \\  src_stride0: u32,
    \\  src_stride1: u32,
    \\  src_stride2: u32,
    \\  src_stride3: u32,
    \\  dst_stride0: u32,
    \\  dst_stride1: u32,
    \\  dst_stride2: u32,
    \\  dst_stride3: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\
    \\fn src_ne(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.src_ne0; }
    \\    case 1u: { return params.src_ne1; }
    \\    case 2u: { return params.src_ne2; }
    \\    case 3u: { return params.src_ne3; }
    \\    default: { return 1u; }
    \\  }
    \\}
    \\
    \\fn src_stride(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.src_stride0; }
    \\    case 1u: { return params.src_stride1; }
    \\    case 2u: { return params.src_stride2; }
    \\    case 3u: { return params.src_stride3; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn dst_stride(dim: u32) -> u32 {
    \\  switch (dim) {
    \\    case 0u: { return params.dst_stride0; }
    \\    case 1u: { return params.dst_stride1; }
    \\    case 2u: { return params.dst_stride2; }
    \\    case 3u: { return params.dst_stride3; }
    \\    default: { return 0u; }
    \\  }
    \\}
    \\
    \\fn repeat_src_index(gid: u32) -> u32 {
    \\  var idx = gid;
    \\  var src_idx = params.src_offset;
    \\  var dim: i32 = 3;
    \\  loop {
    \\    if (dim < 0) {
    \\      break;
    \\    }
    \\    let ud = u32(dim);
    \\    let stride = dst_stride(ud);
    \\    var coord = 0u;
    \\    if (stride != 0u) {
    \\      coord = idx / stride;
    \\      idx = idx % stride;
    \\    }
    \\    let extent = src_ne(ud);
    \\    if (extent != 0u) {
    \\      src_idx = src_idx + (coord % extent) * src_stride(ud);
    \\    }
    \\    dim = dim - 1;
    \\  }
    \\  return src_idx;
    \\}
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let i = gid.x;
    \\  if (i >= params.n) {
    \\    return;
    \\  }
    \\  output.data[params.dst_offset + i] = src.data[repeat_src_index(i)];
    \\}
;
const wgsl_layernorm =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  rows: u32,
    \\  cols: u32,
    \\  eps: f32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\var<workgroup> scratch: array<f32, 256>;
    \\
    \\fn reduce_sum(lid: u32) {
    \\  workgroupBarrier();
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = scratch[lid] + scratch[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\}
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let row = wid.x;
    \\  let lid = lid3.x;
    \\  if (row >= params.rows) {
    \\    return;
    \\  }
    \\  let src_base = params.src_offset + row * params.cols;
    \\  let dst_base = params.dst_offset + row * params.cols;
    \\  var sum = 0.0f;
    \\  var col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    sum = sum + src.data[src_base + col];
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = sum;
    \\  reduce_sum(lid);
    \\  let mean = scratch[0] / f32(params.cols);
    \\  var ss = 0.0f;
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    let diff = src.data[src_base + col] - mean;
    \\    ss = ss + diff * diff;
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = ss;
    \\  reduce_sum(lid);
    \\  let inv_std = 1.0f / sqrt(scratch[0] / f32(params.cols) + params.eps);
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    output.data[dst_base + col] = (src.data[src_base + col] - mean) * inv_std;
    \\    col = col + 256u;
    \\  }
    \\}
;
const wgsl_rmsnorm =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  rows: u32,
    \\  cols: u32,
    \\  eps: f32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\var<workgroup> scratch: array<f32, 256>;
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let row = wid.x;
    \\  let lid = lid3.x;
    \\  if (row >= params.rows) {
    \\    return;
    \\  }
    \\  let src_base = params.src_offset + row * params.cols;
    \\  let dst_base = params.dst_offset + row * params.cols;
    \\  var ss = 0.0f;
    \\  var col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    let v = src.data[src_base + col];
    \\    ss = ss + v * v;
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = ss;
    \\  workgroupBarrier();
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = scratch[lid] + scratch[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let inv_rms = 1.0f / sqrt(scratch[0] / f32(params.cols) + params.eps);
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    output.data[dst_base + col] = src.data[src_base + col] * inv_rms;
    \\    col = col + 256u;
    \\  }
    \\}
;
const wgsl_softmax =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  rows: u32,
    \\  cols: u32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\var<workgroup> scratch: array<f32, 256>;
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let row = wid.x;
    \\  let lid = lid3.x;
    \\  if (row >= params.rows) {
    \\    return;
    \\  }
    \\  let src_base = params.src_offset + row * params.cols;
    \\  let dst_base = params.dst_offset + row * params.cols;
    \\  var row_max = -3.4028234663852886e38f;
    \\  var col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    row_max = max(row_max, src.data[src_base + col]);
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = row_max;
    \\  workgroupBarrier();
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = max(scratch[lid], scratch[lid + stride]);
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let max_value = scratch[0];
    \\  var row_sum = 0.0f;
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    let e = exp(src.data[src_base + col] - max_value);
    \\    output.data[dst_base + col] = e;
    \\    row_sum = row_sum + e;
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = row_sum;
    \\  workgroupBarrier();
    \\  stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = scratch[lid] + scratch[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let inv_sum = 1.0f / scratch[0];
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    output.data[dst_base + col] = output.data[dst_base + col] * inv_sum;
    \\    col = col + 256u;
    \\  }
    \\}
;
const wgsl_logsoftmax =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  rows: u32,
    \\  cols: u32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\var<workgroup> scratch: array<f32, 256>;
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let row = wid.x;
    \\  let lid = lid3.x;
    \\  if (row >= params.rows) {
    \\    return;
    \\  }
    \\  let src_base = params.src_offset + row * params.cols;
    \\  let dst_base = params.dst_offset + row * params.cols;
    \\  var row_max = -3.4028234663852886e38f;
    \\  var col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    row_max = max(row_max, src.data[src_base + col]);
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = row_max;
    \\  workgroupBarrier();
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = max(scratch[lid], scratch[lid + stride]);
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let max_value = scratch[0];
    \\  var row_sum = 0.0f;
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    row_sum = row_sum + exp(src.data[src_base + col] - max_value);
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = row_sum;
    \\  workgroupBarrier();
    \\  stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scratch[lid] = scratch[lid] + scratch[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let log_denom = max_value + log(scratch[0]);
    \\  col = lid;
    \\  loop {
    \\    if (col >= params.cols) {
    \\      break;
    \\    }
    \\    output.data[dst_base + col] = src.data[src_base + col] - log_denom;
    \\    col = col + 256u;
    \\  }
    \\}
;
const wgsl_reduce =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  op: u32,
    \\  n_out: u32,
    \\  reduce_size: u32,
    \\  src_offset: u32,
    \\  dst_offset: u32,
    \\  _pad0: u32,
    \\  _pad1: u32,
    \\  _pad2: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\var<workgroup> scratch: array<f32, 256>;
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let row = wid.x;
    \\  let lid = lid3.x;
    \\  if (row >= params.n_out) {
    \\    return;
    \\  }
    \\  let src_base = params.src_offset + row * params.reduce_size;
    \\  var acc = 0.0f;
    \\  if (params.op == 2u) {
    \\    acc = -3.4028234663852886e38f;
    \\  } else if (params.op == 3u) {
    \\    acc = 3.4028234663852886e38f;
    \\  }
    \\  var col = lid;
    \\  loop {
    \\    if (col >= params.reduce_size) {
    \\      break;
    \\    }
    \\    let value = src.data[src_base + col];
    \\    if (params.op == 2u) {
    \\      acc = max(acc, value);
    \\    } else if (params.op == 3u) {
    \\      acc = min(acc, value);
    \\    } else {
    \\      acc = acc + value;
    \\    }
    \\    col = col + 256u;
    \\  }
    \\  scratch[lid] = acc;
    \\  workgroupBarrier();
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      if (params.op == 2u) {
    \\        scratch[lid] = max(scratch[lid], scratch[lid + stride]);
    \\      } else if (params.op == 3u) {
    \\        scratch[lid] = min(scratch[lid], scratch[lid + stride]);
    \\      } else {
    \\        scratch[lid] = scratch[lid] + scratch[lid + stride];
    \\      }
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  if (lid == 0u) {
    \\    output.data[params.dst_offset + row] = scratch[0];
    \\  }
    \\}
;
const wgsl_rope =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  half_d: u32,
    \\  seq_len: u32,
    \\  src_off: u32,
    \\  cs_off: u32,
    \\  dst_off: u32,
    \\  src_rs: u32,
    \\  src_cs: u32,
    \\  cs_cs: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read> cos_sin: Values;
    \\@group(0) @binding(2) var<storage, read_write> output: Values;
    \\@group(0) @binding(3) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.half_d * params.seq_len;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let col = idx / params.half_d;
    \\  let pair = idx - col * params.half_d;
    \\  let d = params.half_d * 2u;
    \\  let cos_v = cos_sin.data[params.cs_off + col * params.cs_cs + pair];
    \\  let sin_v = cos_sin.data[params.cs_off + col * params.cs_cs + d + pair];
    \\  let x_lo = src.data[params.src_off + col * params.src_cs + pair * params.src_rs];
    \\  let x_hi = src.data[params.src_off + col * params.src_cs + (pair + params.half_d) * params.src_rs];
    \\  let dst_base = params.dst_off + col * d;
    \\  output.data[dst_base + pair] = x_lo * cos_v - x_hi * sin_v;
    \\  output.data[dst_base + pair + params.half_d] = x_hi * cos_v + x_lo * sin_v;
    \\}
;
const wgsl_slice_assign =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  rows: u32,
    \\  cols: u32,
    \\  dst_offset: u32,
    \\  dst_row_stride: u32,
    \\  dst_col_stride: u32,
    \\  src_offset: u32,
    \\  src_row_stride: u32,
    \\  src_col_stride: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> src: Values;
    \\@group(0) @binding(1) var<storage, read_write> output: Values;
    \\@group(0) @binding(2) var<uniform> params: Params;
    \\
    \\@compute @workgroup_size(64)
    \\fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    \\  let idx = gid.x;
    \\  let total = params.rows * params.cols;
    \\  if (idx >= total) {
    \\    return;
    \\  }
    \\  let row = idx % params.rows;
    \\  let col = idx / params.rows;
    \\  let src_idx = params.src_offset + row * params.src_row_stride + col * params.src_col_stride;
    \\  let dst_idx = params.dst_offset + row * params.dst_row_stride + col * params.dst_col_stride;
    \\  output.data[dst_idx] = src.data[src_idx];
    \\}
;
const wgsl_attention =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  d_head: u32,
    \\  seq_q: u32,
    \\  seq_kv: u32,
    \\  has_mask: u32,
    \\  scale: f32,
    \\  q_off: u32,
    \\  k_off: u32,
    \\  v_off: u32,
    \\  mask_off: u32,
    \\  dst_off: u32,
    \\  q_rs: u32,
    \\  q_cs: u32,
    \\  k_rs: u32,
    \\  k_cs: u32,
    \\  v_rs: u32,
    \\  v_cs: u32,
    \\  mask_rs: u32,
    \\  mask_cs: u32,
    \\  dst_rs: u32,
    \\  dst_cs: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> q: Values;
    \\@group(0) @binding(1) var<storage, read> k: Values;
    \\@group(0) @binding(2) var<storage, read> v: Values;
    \\@group(0) @binding(3) var<storage, read> mask: Values;
    \\@group(0) @binding(4) var<storage, read_write> output: Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\var<workgroup> scores: array<f32, 256>;
    \\var<workgroup> weights: array<f32, 256>;
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let qi = wid.x;
    \\  let lid = lid3.x;
    \\  if (qi >= params.seq_q) {
    \\    return;
    \\  }
    \\
    \\  var score = -3.4028234663852886e38f;
    \\  if (lid < params.seq_kv) {
    \\    var dot = 0.0f;
    \\    var r = 0u;
    \\    loop {
    \\      if (r >= params.d_head) {
    \\        break;
    \\      }
    \\      let q_idx = params.q_off + qi * params.q_cs + r * params.q_rs;
    \\      let k_idx = params.k_off + lid * params.k_cs + r * params.k_rs;
    \\      dot = dot + q.data[q_idx] * k.data[k_idx];
    \\      r = r + 1u;
    \\    }
    \\    var mask_add = 0.0f;
    \\    if (params.has_mask != 0u) {
    \\      mask_add = mask.data[params.mask_off + qi * params.mask_cs + lid * params.mask_rs];
    \\    }
    \\    score = dot * params.scale + mask_add;
    \\  }
    \\  scores[lid] = score;
    \\  workgroupBarrier();
    \\
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scores[lid] = max(scores[lid], scores[lid + stride]);
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let max_score = scores[0];
    \\  var weight = 0.0f;
    \\  if (lid < params.seq_kv) {
    \\    weight = exp(score - max_score);
    \\  }
    \\  weights[lid] = weight;
    \\  scores[lid] = weight;
    \\  workgroupBarrier();
    \\
    \\  stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      scores[lid] = scores[lid] + scores[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  var inv_sum = 0.0f;
    \\  if (scores[0] > 0.0f) {
    \\    inv_sum = 1.0f / scores[0];
    \\  }
    \\
    \\  var out_col = lid;
    \\  loop {
    \\    if (out_col >= params.d_head) {
    \\      break;
    \\    }
    \\    var acc = 0.0f;
    \\    var s = 0u;
    \\    loop {
    \\      if (s >= params.seq_kv) {
    \\        break;
    \\      }
    \\      let v_idx = params.v_off + s * params.v_cs + out_col * params.v_rs;
    \\      acc = acc + weights[s] * v.data[v_idx];
    \\      s = s + 1u;
    \\    }
    \\    let dst_idx = params.dst_off + qi * params.dst_cs + out_col * params.dst_rs;
    \\    output.data[dst_idx] = acc * inv_sum;
    \\    out_col = out_col + 256u;
    \\  }
    \\}
;
const wgsl_attention_long =
    \\struct Values {
    \\  data: array<f32>,
    \\};
    \\struct Params {
    \\  d_head: u32,
    \\  seq_q: u32,
    \\  seq_kv: u32,
    \\  has_mask: u32,
    \\  scale: f32,
    \\  q_off: u32,
    \\  k_off: u32,
    \\  v_off: u32,
    \\  mask_off: u32,
    \\  dst_off: u32,
    \\  q_rs: u32,
    \\  q_cs: u32,
    \\  k_rs: u32,
    \\  k_cs: u32,
    \\  v_rs: u32,
    \\  v_cs: u32,
    \\  mask_rs: u32,
    \\  mask_cs: u32,
    \\  dst_rs: u32,
    \\  dst_cs: u32,
    \\};
    \\@group(0) @binding(0) var<storage, read> q: Values;
    \\@group(0) @binding(1) var<storage, read> k: Values;
    \\@group(0) @binding(2) var<storage, read> v: Values;
    \\@group(0) @binding(3) var<storage, read> mask: Values;
    \\@group(0) @binding(4) var<storage, read_write> output: Values;
    \\@group(0) @binding(5) var<uniform> params: Params;
    \\var<workgroup> partials: array<f32, 256>;
    \\
    \\fn attention_score(qi: u32, key: u32) -> f32 {
    \\  var dot = 0.0f;
    \\  var r = 0u;
    \\  loop {
    \\    if (r >= params.d_head) {
    \\      break;
    \\    }
    \\    let q_idx = params.q_off + qi * params.q_cs + r * params.q_rs;
    \\    let k_idx = params.k_off + key * params.k_cs + r * params.k_rs;
    \\    dot = dot + q.data[q_idx] * k.data[k_idx];
    \\    r = r + 1u;
    \\  }
    \\  var mask_add = 0.0f;
    \\  if (params.has_mask != 0u) {
    \\    mask_add = mask.data[params.mask_off + qi * params.mask_cs + key * params.mask_rs];
    \\  }
    \\  return dot * params.scale + mask_add;
    \\}
    \\
    \\@compute @workgroup_size(256)
    \\fn main(
    \\  @builtin(workgroup_id) wid: vec3<u32>,
    \\  @builtin(local_invocation_id) lid3: vec3<u32>,
    \\) {
    \\  let qi = wid.x;
    \\  let lid = lid3.x;
    \\  if (qi >= params.seq_q) {
    \\    return;
    \\  }
    \\
    \\  var local_max = -3.4028234663852886e38f;
    \\  var key = lid;
    \\  loop {
    \\    if (key >= params.seq_kv) {
    \\      break;
    \\    }
    \\    local_max = max(local_max, attention_score(qi, key));
    \\    key = key + 256u;
    \\  }
    \\  partials[lid] = local_max;
    \\  workgroupBarrier();
    \\
    \\  var stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      partials[lid] = max(partials[lid], partials[lid + stride]);
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  let max_score = partials[0];
    \\
    \\  var local_sum = 0.0f;
    \\  key = lid;
    \\  loop {
    \\    if (key >= params.seq_kv) {
    \\      break;
    \\    }
    \\    local_sum = local_sum + exp(attention_score(qi, key) - max_score);
    \\    key = key + 256u;
    \\  }
    \\  partials[lid] = local_sum;
    \\  workgroupBarrier();
    \\
    \\  stride = 128u;
    \\  loop {
    \\    if (stride == 0u) {
    \\      break;
    \\    }
    \\    if (lid < stride) {
    \\      partials[lid] = partials[lid] + partials[lid + stride];
    \\    }
    \\    workgroupBarrier();
    \\    stride = stride / 2u;
    \\  }
    \\  var inv_sum = 0.0f;
    \\  if (partials[0] > 0.0f) {
    \\    inv_sum = 1.0f / partials[0];
    \\  }
    \\
    \\  var out_col = lid;
    \\  loop {
    \\    if (out_col >= params.d_head) {
    \\      break;
    \\    }
    \\    var acc = 0.0f;
    \\    var s = 0u;
    \\    loop {
    \\      if (s >= params.seq_kv) {
    \\        break;
    \\      }
    \\      let weight = exp(attention_score(qi, s) - max_score);
    \\      let v_idx = params.v_off + s * params.v_cs + out_col * params.v_rs;
    \\      acc = acc + weight * v.data[v_idx];
    \\      s = s + 1u;
    \\    }
    \\    let dst_idx = params.dst_off + qi * params.dst_cs + out_col * params.dst_rs;
    \\    output.data[dst_idx] = acc * inv_sum;
    \\    out_col = out_col + 256u;
    \\  }
    \\}
;

pub const WgpuBackend = struct {
    alloc: std.mem.Allocator = std.heap.page_allocator,
    shared_gpu: ?GpuContext = null,

    pub fn init(alloc: std.mem.Allocator) WgpuBackend {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *WgpuBackend) void {
        if (self.shared_gpu) |*gpu| gpu.deinit();
        self.shared_gpu = null;
    }

    pub fn backend(self: *WgpuBackend) backend_mod.Backend {
        return .{
            .ctx = self,
            .vtable = &vtable,
            .name_str = "wgpu",
            .device_type = .webgpu,
            .capabilities = tinyLinearCapabilities(),
        };
    }

    fn acquireGpu(self: *WgpuBackend) !GpuContext {
        if (self.shared_gpu == null) {
            self.shared_gpu = try GpuContext.init();
        }
        return self.shared_gpu.?.retain();
    }
};

fn tinyLinearCapabilities() backend_mod.Capabilities {
    var caps = backend_mod.Capabilities.webgpu;
    caps.qmatmul = true;
    caps.runtime_qweights = true;
    caps.softmax = true;
    caps.logsoftmax = true;
    caps.layernorm = true;
    caps.rmsnorm = true;
    caps.reduce = true;
    caps.repeat = true;
    caps.slice_assign = true;
    caps.rope = true;
    caps.external_resources = true;
    caps.fused_elementwise = true;
    caps.max_fused_elementwise_steps = @intCast(max_fused_elementwise_steps);
    caps.attention = .{ .supported = true, .max_seq_kv = max_attention_seq_kv, .max_d_head = max_attention_d_head };
    return caps;
}

const TinyLinearShape = struct {
    input: u16,
    weights: u16,
    bias: u16,
    output: u16,
    temp: u16,
    k: u32,
    n: u32,
};

const MatmulShape = struct {
    a: u16,
    b: u16,
    output: u16,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    dst_row_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    cells: u32,
};

const MatmulElementwiseShape = struct {
    a: u16,
    b: u16,
    secondary: u16,
    output: u16,
    op_code: u32,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
    cells: u32,
};

const MatmulFusedElementwiseShape = struct {
    a: u16,
    b: u16,
    output: u16,
    secondary_bufs: [max_matmul_fused_elementwise_secondaries]u16,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    dst_row_stride: u32,
    a_offset: u32,
    b_offset: u32,
    primary_dst_offset: u32,
    ew_dst_offset: u32,
    n_steps: u32,
    cells: u32,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const MatvecElementwiseShape = struct {
    input: u16,
    weights: u16,
    secondary: u16,
    output: u16,
    op_code: u32,
    n: u32,
    k: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
};

const MatvecFusedElementwiseShape = struct {
    input: u16,
    weights: u16,
    output: u16,
    secondary_bufs: [max_matvec_fused_elementwise_secondaries]u16,
    n: u32,
    k: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    n_steps: u32,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const MatvecSliceShape = struct {
    input: u16,
    weights: u16,
    output: u16,
    n: u32,
    k: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    slice_src_col_start: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
    slice_rows: u32,
    cells: u32,
};

const MatvecRopeSliceShape = struct {
    input: u16,
    weights: u16,
    cos_sin: u16,
    output: u16,
    n: u32,
    k: u32,
    half_d: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    rope_src_col_start: u32,
    cs_off: u32,
    cs_cs: u32,
    rope_dst_offset: u32,
    rope_dst_row_stride: u32,
    rope_dst_col_stride: u32,
};

const QMatmulShape = struct {
    input: u16,
    output: u16,
    qweight_idx: u16,
    m: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    cells: u32,
};

const QMatmulElementwiseShape = struct {
    input: u16,
    secondary: u16,
    output: u16,
    qweight_idx: u16,
    op_code: u32,
    m: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
    cells: u32,
};

const QMatvecSliceShape = struct {
    input: u16,
    output: u16,
    qweight_idx: u16,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    slice_src_col_start: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
    slice_rows: u32,
    cells: u32,
};

const QMatvecRopeSliceShape = struct {
    input: u16,
    cos_sin: u16,
    output: u16,
    qweight_idx: u16,
    n: u32,
    k: u32,
    block_size: u32,
    half_d: u32,
    input_offset: u32,
    rope_src_col_start: u32,
    cs_off: u32,
    cs_cs: u32,
    rope_dst_offset: u32,
    rope_dst_row_stride: u32,
    rope_dst_col_stride: u32,
};

const QMatvecElementwiseShape = struct {
    input: u16,
    secondary: u16,
    output: u16,
    qweight_idx: u16,
    op_code: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
};

const QMatvecFusedElementwiseShape = struct {
    input: u16,
    output: u16,
    qweight_idx: u16,
    secondary_bufs: [max_qmatvec_fused_elementwise_secondaries]u16,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    dst_offset: u32,
    n_steps: u32,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const QLinearShape = struct {
    input: u16,
    bias: u16,
    output: u16,
    qweight_idx: u16,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    bias_offset: u32,
    dst_offset: u32,
};

const ElementwiseShape = struct {
    op_code: u32,
    src0: u16,
    src1: u16,
    output: u16,
    n: u32,
    dst_offset: u32,
    src0_offset: u32,
    src1_offset: u32,
};

const FusedElementwiseShape = struct {
    src: u16,
    output: u16,
    secondary_bufs: [max_fused_elementwise_secondaries]u16,
    n: u32,
    n_steps: u32,
    dst_offset: u32,
    src_offset: u32,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_repeat: [max_fused_elementwise_steps]u32,
    secondary_repeat_dst_offset: [max_fused_elementwise_steps]u32,
    repeat_src_offset: u32,
    repeat_src_ne: [4]u32,
    repeat_src_strides: [4]u32,
    repeat_dst_strides: [4]u32,
};

const RepeatShape = struct {
    src: u16,
    output: u16,
    n: u32,
    src_offset: u32,
    dst_offset: u32,
    src_ne: [4]u32,
    src_strides: [4]u32,
    dst_strides: [4]u32,
};

const LayerNormShape = struct {
    src: u16,
    output: u16,
    rows: u32,
    cols: u32,
    eps: f32,
    src_offset: u32,
    dst_offset: u32,
};

const RmsNormShape = struct {
    src: u16,
    output: u16,
    rows: u32,
    cols: u32,
    eps: f32,
    src_offset: u32,
    dst_offset: u32,
};

const SoftmaxShape = struct {
    src: u16,
    output: u16,
    rows: u32,
    cols: u32,
    src_offset: u32,
    dst_offset: u32,
};

const ReduceShape = struct {
    op_code: u32,
    src: u16,
    output: u16,
    n_out: u32,
    reduce_size: u32,
    src_offset: u32,
    dst_offset: u32,
};

const RopeShape = struct {
    src: u16,
    cos_sin: u16,
    output: u16,
    half_d: u32,
    seq_len: u32,
    src_off: u32,
    cs_off: u32,
    dst_off: u32,
    src_rs: u32,
    src_cs: u32,
    cs_cs: u32,
    cells: u32,
};

const SliceAssignShape = struct {
    src: u16,
    output: u16,
    rows: u32,
    cols: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    dst_col_stride: u32,
    src_offset: u32,
    src_row_stride: u32,
    src_col_stride: u32,
    cells: u32,
};

const AttentionShape = struct {
    q: u16,
    k: u16,
    v: u16,
    mask: u16,
    output: u16,
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    has_mask: u32,
    scale: f32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    mask_off: u32,
    dst_off: u32,
    q_rs: u32,
    q_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
};

const ProgramShape = union(enum) {
    tiny_linear: TinyLinearShape,
    matmul: MatmulShape,
    matmul_elementwise: MatmulElementwiseShape,
    matmul_fused_elementwise: MatmulFusedElementwiseShape,
    matvec_elementwise: MatvecElementwiseShape,
    matvec_fused_elementwise: MatvecFusedElementwiseShape,
    matvec_slice: MatvecSliceShape,
    matvec_rope_slice: MatvecRopeSliceShape,
    qmatmul: QMatmulShape,
    qmatmul_elementwise: QMatmulElementwiseShape,
    qmatvec_slice: QMatvecSliceShape,
    qmatvec_rope_slice: QMatvecRopeSliceShape,
    qmatvec_elementwise: QMatvecElementwiseShape,
    qmatvec_fused_elementwise: QMatvecFusedElementwiseShape,
    qlinear: QLinearShape,
    elementwise: ElementwiseShape,
    fused_elementwise: FusedElementwiseShape,
    repeat: RepeatShape,
    layernorm: LayerNormShape,
    rmsnorm: RmsNormShape,
    softmax: SoftmaxShape,
    logsoftmax: SoftmaxShape,
    reduce: ReduceShape,
    rope: RopeShape,
    slice_assign: SliceAssignShape,
    attention: AttentionShape,

    fn shaderSource(self: ProgramShape) []const u8 {
        return switch (self) {
            .tiny_linear => wgsl_linear,
            .matmul => wgsl_matmul,
            .matmul_elementwise => wgsl_matmul_elementwise,
            .matmul_fused_elementwise => wgsl_matmul_fused_elementwise,
            .matvec_elementwise => wgsl_matvec_elementwise,
            .matvec_fused_elementwise => wgsl_matvec_fused_elementwise,
            .matvec_slice => wgsl_matvec_slice,
            .matvec_rope_slice => wgsl_matvec_rope_slice,
            .qmatmul => wgsl_qmatmul,
            .qmatmul_elementwise => wgsl_qmatmul_elementwise,
            .qmatvec_slice => wgsl_qmatvec_slice,
            .qmatvec_rope_slice => wgsl_qmatvec_rope_slice,
            .qmatvec_elementwise => wgsl_qmatvec_elementwise,
            .qmatvec_fused_elementwise => wgsl_qmatvec_fused_elementwise,
            .qlinear => wgsl_qlinear,
            .elementwise => wgsl_elementwise,
            .fused_elementwise => wgsl_fused_elementwise,
            .repeat => wgsl_repeat,
            .layernorm => wgsl_layernorm,
            .rmsnorm => wgsl_rmsnorm,
            .softmax => wgsl_softmax,
            .logsoftmax => wgsl_logsoftmax,
            .reduce => wgsl_reduce,
            .rope => wgsl_rope,
            .slice_assign => wgsl_slice_assign,
            .attention => |shape| if (shape.seq_kv <= 256) wgsl_attention else wgsl_attention_long,
        };
    }

    fn paramsByteLen(self: ProgramShape) u64 {
        return switch (self) {
            .tiny_linear => @sizeOf(LinearParams),
            .matmul => @sizeOf(MatmulParams),
            .matmul_elementwise => @sizeOf(MatmulElementwiseParams),
            .matmul_fused_elementwise => @sizeOf(MatmulFusedElementwiseParams),
            .matvec_elementwise => @sizeOf(MatvecElementwiseParams),
            .matvec_fused_elementwise => @sizeOf(MatvecFusedElementwiseParams),
            .matvec_slice => @sizeOf(MatvecSliceParams),
            .matvec_rope_slice => @sizeOf(MatvecRopeSliceParams),
            .qmatmul => @sizeOf(QMatmulParams),
            .qmatmul_elementwise => @sizeOf(QMatmulElementwiseParams),
            .qmatvec_slice => @sizeOf(QMatvecSliceParams),
            .qmatvec_rope_slice => @sizeOf(QMatvecRopeSliceParams),
            .qmatvec_elementwise => @sizeOf(QMatvecElementwiseParams),
            .qmatvec_fused_elementwise => @sizeOf(QMatvecFusedElementwiseParams),
            .qlinear => @sizeOf(QLinearParams),
            .elementwise => @sizeOf(ElementwiseParams),
            .fused_elementwise => @sizeOf(FusedElementwiseParams),
            .repeat => @sizeOf(RepeatParams),
            .layernorm => @sizeOf(LayerNormParams),
            .rmsnorm => @sizeOf(RmsNormParams),
            .softmax => @sizeOf(SoftmaxParams),
            .logsoftmax => @sizeOf(SoftmaxParams),
            .reduce => @sizeOf(ReduceParams),
            .rope => @sizeOf(RopeParams),
            .slice_assign => @sizeOf(SliceAssignParams),
            .attention => @sizeOf(AttentionParams),
        };
    }

    fn outputBuffer(self: ProgramShape) u16 {
        return switch (self) {
            .tiny_linear => |shape| shape.output,
            .matmul => |shape| shape.output,
            .matmul_elementwise => |shape| shape.output,
            .matmul_fused_elementwise => |shape| shape.output,
            .matvec_elementwise => |shape| shape.output,
            .matvec_fused_elementwise => |shape| shape.output,
            .matvec_slice => |shape| shape.output,
            .matvec_rope_slice => |shape| shape.output,
            .qmatmul => |shape| shape.output,
            .qmatmul_elementwise => |shape| shape.output,
            .qmatvec_slice => |shape| shape.output,
            .qmatvec_rope_slice => |shape| shape.output,
            .qmatvec_elementwise => |shape| shape.output,
            .qmatvec_fused_elementwise => |shape| shape.output,
            .qlinear => |shape| shape.output,
            .elementwise => |shape| shape.output,
            .fused_elementwise => |shape| shape.output,
            .repeat => |shape| shape.output,
            .layernorm => |shape| shape.output,
            .rmsnorm => |shape| shape.output,
            .softmax => |shape| shape.output,
            .logsoftmax => |shape| shape.output,
            .reduce => |shape| shape.output,
            .rope => |shape| shape.output,
            .slice_assign => |shape| shape.output,
            .attention => |shape| shape.output,
        };
    }

    fn dispatchWorkgroups(self: ProgramShape) u32 {
        const invocations: u32 = switch (self) {
            .tiny_linear => |shape| shape.n,
            .matmul => |shape| shape.cells,
            .matmul_elementwise => |shape| shape.cells,
            .matmul_fused_elementwise => |shape| shape.cells,
            .matvec_elementwise => |shape| shape.n,
            .matvec_fused_elementwise => |shape| shape.n,
            .matvec_slice => |shape| shape.cells,
            .matvec_rope_slice => |shape| shape.half_d,
            .qmatmul => |shape| shape.cells,
            .qmatmul_elementwise => |shape| shape.cells,
            .qmatvec_slice => |shape| shape.cells,
            .qmatvec_rope_slice => |shape| shape.half_d,
            .qmatvec_elementwise => |shape| shape.n,
            .qmatvec_fused_elementwise => |shape| shape.n,
            .qlinear => |shape| shape.n,
            .elementwise => |shape| shape.n,
            .fused_elementwise => |shape| shape.n,
            .repeat => |shape| shape.n,
            .layernorm => |shape| shape.rows,
            .rmsnorm => |shape| shape.rows,
            .softmax => |shape| shape.rows,
            .logsoftmax => |shape| shape.rows,
            .reduce => |shape| shape.n_out,
            .rope => |shape| shape.cells,
            .slice_assign => |shape| shape.cells,
            .attention => |shape| shape.seq_q,
        };
        return switch (self) {
            .layernorm, .rmsnorm, .softmax, .logsoftmax, .reduce, .attention => invocations,
            else => invocations / 64 + @intFromBool(invocations % 64 != 0),
        };
    }

    fn writeParams(self: ProgramShape, queue: c.WGPUQueue, buffer: c.WGPUBuffer) void {
        switch (self) {
            .tiny_linear => |shape| {
                const params = LinearParams{ .k = shape.k, .n = shape.n };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(LinearParams));
            },
            .matmul => |shape| {
                const params = MatmulParams{
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .a_row_stride = shape.a_row_stride,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .dst_row_stride = shape.dst_row_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatmulParams));
            },
            .matmul_elementwise => |shape| {
                const params = MatmulElementwiseParams{
                    .op = shape.op_code,
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .a_row_stride = shape.a_row_stride,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .secondary_offset = shape.secondary_offset,
                    .dst_offset = shape.dst_offset,
                    .is_swapped = shape.is_swapped,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatmulElementwiseParams));
            },
            .matmul_fused_elementwise => |shape| {
                const params = MatmulFusedElementwiseParams{
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .a_row_stride = shape.a_row_stride,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .dst_row_stride = shape.dst_row_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .primary_dst_offset = shape.primary_dst_offset,
                    .ew_dst_offset = shape.ew_dst_offset,
                    .n_steps = shape.n_steps,
                    .op = shape.op,
                    .is_swapped = shape.is_swapped,
                    .secondary_slot = shape.secondary_slot,
                    .secondary_offset = shape.secondary_offset,
                    .secondary_is_primary = shape.secondary_is_primary,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatmulFusedElementwiseParams));
            },
            .matvec_elementwise => |shape| {
                const params = MatvecElementwiseParams{
                    .op = shape.op_code,
                    .n = shape.n,
                    .k = shape.k,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .secondary_offset = shape.secondary_offset,
                    .dst_offset = shape.dst_offset,
                    .is_swapped = shape.is_swapped,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatvecElementwiseParams));
            },
            .matvec_fused_elementwise => |shape| {
                const params = MatvecFusedElementwiseParams{
                    .n = shape.n,
                    .k = shape.k,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .dst_offset = shape.dst_offset,
                    .n_steps = shape.n_steps,
                    .op = shape.op,
                    .is_swapped = shape.is_swapped,
                    .secondary_slot = shape.secondary_slot,
                    .secondary_offset = shape.secondary_offset,
                    .secondary_is_primary = shape.secondary_is_primary,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatvecFusedElementwiseParams));
            },
            .matvec_slice => |shape| {
                const params = MatvecSliceParams{
                    .n = shape.n,
                    .k = shape.k,
                    .cells = shape.cells,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .slice_src_col_start = shape.slice_src_col_start,
                    .slice_dst_offset = shape.slice_dst_offset,
                    .slice_dst_row_stride = shape.slice_dst_row_stride,
                    .slice_dst_col_stride = shape.slice_dst_col_stride,
                    .slice_rows = shape.slice_rows,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatvecSliceParams));
            },
            .matvec_rope_slice => |shape| {
                const params = MatvecRopeSliceParams{
                    .n = shape.n,
                    .k = shape.k,
                    .half_d = shape.half_d,
                    .a_col_stride = shape.a_col_stride,
                    .b_row_stride = shape.b_row_stride,
                    .b_col_stride = shape.b_col_stride,
                    .a_offset = shape.a_offset,
                    .b_offset = shape.b_offset,
                    .rope_src_col_start = shape.rope_src_col_start,
                    .cs_off = shape.cs_off,
                    .cs_cs = shape.cs_cs,
                    .rope_dst_offset = shape.rope_dst_offset,
                    .rope_dst_row_stride = shape.rope_dst_row_stride,
                    .rope_dst_col_stride = shape.rope_dst_col_stride,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(MatvecRopeSliceParams));
            },
            .qmatmul => |shape| {
                const params = QMatmulParams{
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .input_offset = shape.input_offset,
                    .input_row_stride = shape.input_row_stride,
                    .dst_offset = shape.dst_offset,
                    .dst_row_stride = shape.dst_row_stride,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatmulParams));
            },
            .qmatmul_elementwise => |shape| {
                const params = QMatmulElementwiseParams{
                    .op = shape.op_code,
                    .m = shape.m,
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .input_offset = shape.input_offset,
                    .input_row_stride = shape.input_row_stride,
                    .secondary_offset = shape.secondary_offset,
                    .dst_offset = shape.dst_offset,
                    .is_swapped = shape.is_swapped,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatmulElementwiseParams));
            },
            .qmatvec_slice => |shape| {
                const params = QMatvecSliceParams{
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .cells = shape.cells,
                    .input_offset = shape.input_offset,
                    .slice_src_col_start = shape.slice_src_col_start,
                    .slice_dst_offset = shape.slice_dst_offset,
                    .slice_dst_row_stride = shape.slice_dst_row_stride,
                    .slice_dst_col_stride = shape.slice_dst_col_stride,
                    .slice_rows = shape.slice_rows,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatvecSliceParams));
            },
            .qmatvec_rope_slice => |shape| {
                const params = QMatvecRopeSliceParams{
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .half_d = shape.half_d,
                    .input_offset = shape.input_offset,
                    .rope_src_col_start = shape.rope_src_col_start,
                    .cs_off = shape.cs_off,
                    .cs_cs = shape.cs_cs,
                    .rope_dst_offset = shape.rope_dst_offset,
                    .rope_dst_row_stride = shape.rope_dst_row_stride,
                    .rope_dst_col_stride = shape.rope_dst_col_stride,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatvecRopeSliceParams));
            },
            .qmatvec_elementwise => |shape| {
                const params = QMatvecElementwiseParams{
                    .op = shape.op_code,
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .input_offset = shape.input_offset,
                    .secondary_offset = shape.secondary_offset,
                    .dst_offset = shape.dst_offset,
                    .is_swapped = shape.is_swapped,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatvecElementwiseParams));
            },
            .qmatvec_fused_elementwise => |shape| {
                const params = QMatvecFusedElementwiseParams{
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .input_offset = shape.input_offset,
                    .dst_offset = shape.dst_offset,
                    .n_steps = shape.n_steps,
                    .op = shape.op,
                    .is_swapped = shape.is_swapped,
                    .secondary_slot = shape.secondary_slot,
                    .secondary_offset = shape.secondary_offset,
                    .secondary_is_primary = shape.secondary_is_primary,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QMatvecFusedElementwiseParams));
            },
            .qlinear => |shape| {
                const params = QLinearParams{
                    .n = shape.n,
                    .k = shape.k,
                    .block_size = shape.block_size,
                    .input_offset = shape.input_offset,
                    .bias_offset = shape.bias_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(QLinearParams));
            },
            .elementwise => |shape| {
                const params = ElementwiseParams{
                    .op = shape.op_code,
                    .n = shape.n,
                    .dst_offset = shape.dst_offset,
                    .src0_offset = shape.src0_offset,
                    .src1_offset = shape.src1_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(ElementwiseParams));
            },
            .fused_elementwise => |shape| {
                const params = FusedElementwiseParams{
                    .n = shape.n,
                    .n_steps = shape.n_steps,
                    .dst_offset = shape.dst_offset,
                    .src_offset = shape.src_offset,
                    .op = shape.op,
                    .is_swapped = shape.is_swapped,
                    .secondary_slot = shape.secondary_slot,
                    .secondary_offset = shape.secondary_offset,
                    .secondary_is_repeat = shape.secondary_is_repeat,
                    .secondary_repeat_dst_offset = shape.secondary_repeat_dst_offset,
                    .repeat_src_offset = shape.repeat_src_offset,
                    .repeat_src_ne = shape.repeat_src_ne,
                    .repeat_src_strides = shape.repeat_src_strides,
                    .repeat_dst_strides = shape.repeat_dst_strides,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(FusedElementwiseParams));
            },
            .repeat => |shape| {
                const params = RepeatParams{
                    .n = shape.n,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                    .src_ne = shape.src_ne,
                    .src_strides = shape.src_strides,
                    .dst_strides = shape.dst_strides,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(RepeatParams));
            },
            .layernorm => |shape| {
                const params = LayerNormParams{
                    .rows = shape.rows,
                    .cols = shape.cols,
                    .eps = shape.eps,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(LayerNormParams));
            },
            .rmsnorm => |shape| {
                const params = RmsNormParams{
                    .rows = shape.rows,
                    .cols = shape.cols,
                    .eps = shape.eps,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(RmsNormParams));
            },
            .softmax => |shape| {
                const params = SoftmaxParams{
                    .rows = shape.rows,
                    .cols = shape.cols,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(SoftmaxParams));
            },
            .logsoftmax => |shape| {
                const params = SoftmaxParams{
                    .rows = shape.rows,
                    .cols = shape.cols,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(SoftmaxParams));
            },
            .reduce => |shape| {
                const params = ReduceParams{
                    .op = shape.op_code,
                    .n_out = shape.n_out,
                    .reduce_size = shape.reduce_size,
                    .src_offset = shape.src_offset,
                    .dst_offset = shape.dst_offset,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(ReduceParams));
            },
            .rope => |shape| {
                const params = RopeParams{
                    .half_d = shape.half_d,
                    .seq_len = shape.seq_len,
                    .src_off = shape.src_off,
                    .cs_off = shape.cs_off,
                    .dst_off = shape.dst_off,
                    .src_rs = shape.src_rs,
                    .src_cs = shape.src_cs,
                    .cs_cs = shape.cs_cs,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(RopeParams));
            },
            .slice_assign => |shape| {
                const params = SliceAssignParams{
                    .rows = shape.rows,
                    .cols = shape.cols,
                    .dst_offset = shape.dst_offset,
                    .dst_row_stride = shape.dst_row_stride,
                    .dst_col_stride = shape.dst_col_stride,
                    .src_offset = shape.src_offset,
                    .src_row_stride = shape.src_row_stride,
                    .src_col_stride = shape.src_col_stride,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(SliceAssignParams));
            },
            .attention => |shape| {
                const params = AttentionParams{
                    .d_head = shape.d_head,
                    .seq_q = shape.seq_q,
                    .seq_kv = shape.seq_kv,
                    .has_mask = shape.has_mask,
                    .scale = shape.scale,
                    .q_off = shape.q_off,
                    .k_off = shape.k_off,
                    .v_off = shape.v_off,
                    .mask_off = shape.mask_off,
                    .dst_off = shape.dst_off,
                    .q_rs = shape.q_rs,
                    .q_cs = shape.q_cs,
                    .k_rs = shape.k_rs,
                    .k_cs = shape.k_cs,
                    .v_rs = shape.v_rs,
                    .v_cs = shape.v_cs,
                    .mask_rs = shape.mask_rs,
                    .mask_cs = shape.mask_cs,
                    .dst_rs = shape.dst_rs,
                    .dst_cs = shape.dst_cs,
                };
                c.wgpuQueueWriteBuffer(queue, buffer, 0, &params, @sizeOf(AttentionParams));
            },
        }
    }

    fn writeRuntimeParams(self: ProgramShape, queue: c.WGPUQueue, buffer: c.WGPUBuffer, ops: []const backend_mod.DeviceOp) void {
        switch (self) {
            .slice_assign => |shape| {
                var runtime_shape = shape;
                if (ops.len == 1 and ops[0] == .slice_assign) {
                    runtime_shape.dst_offset = ops[0].slice_assign.dst_offset;
                }
                (ProgramShape{ .slice_assign = runtime_shape }).writeParams(queue, buffer);
            },
            .matvec_slice => |shape| {
                var runtime_shape = shape;
                if (ops.len == 2 and ops[1] == .slice_assign) {
                    runtime_shape.slice_dst_offset = ops[1].slice_assign.dst_offset;
                }
                (ProgramShape{ .matvec_slice = runtime_shape }).writeParams(queue, buffer);
            },
            .matvec_rope_slice => |shape| {
                var runtime_shape = shape;
                if (ops.len == 3 and ops[2] == .slice_assign) {
                    runtime_shape.rope_dst_offset = ops[2].slice_assign.dst_offset;
                }
                (ProgramShape{ .matvec_rope_slice = runtime_shape }).writeParams(queue, buffer);
            },
            .qmatvec_slice => |shape| {
                var runtime_shape = shape;
                if (ops.len == 2 and ops[1] == .slice_assign) {
                    runtime_shape.slice_dst_offset = ops[1].slice_assign.dst_offset;
                }
                (ProgramShape{ .qmatvec_slice = runtime_shape }).writeParams(queue, buffer);
            },
            .qmatvec_rope_slice => |shape| {
                var runtime_shape = shape;
                if (ops.len == 3 and ops[2] == .slice_assign) {
                    runtime_shape.rope_dst_offset = ops[2].slice_assign.dst_offset;
                }
                (ProgramShape{ .qmatvec_rope_slice = runtime_shape }).writeParams(queue, buffer);
            },
            .attention => |shape| {
                var runtime_shape = shape;
                if (ops.len == 1 and ops[0] == .attention) {
                    runtime_shape.seq_kv = ops[0].attention.seq_kv;
                }
                (ProgramShape{ .attention = runtime_shape }).writeParams(queue, buffer);
            },
            else => self.writeParams(queue, buffer),
        }
    }
};

const GpuContext = struct {
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,

    fn init() !GpuContext {
        var instance_desc: c.WGPUInstanceDescriptor = std.mem.zeroes(c.WGPUInstanceDescriptor);
        const instance = c.wgpuCreateInstance(&instance_desc) orelse return error.WgpuUnavailable;
        errdefer c.wgpuInstanceRelease(instance);

        var adapter_state = AdapterRequest{};
        var adapter_cb: c.WGPURequestAdapterCallbackInfo = std.mem.zeroes(c.WGPURequestAdapterCallbackInfo);
        adapter_cb.mode = c.WGPUCallbackMode_AllowProcessEvents;
        adapter_cb.callback = requestAdapterCallback;
        adapter_cb.userdata1 = &adapter_state;
        var adapter_options: c.WGPURequestAdapterOptions = std.mem.zeroes(c.WGPURequestAdapterOptions);
        _ = c.wgpuInstanceRequestAdapter(instance, &adapter_options, adapter_cb);
        if (!waitForCallback(instance, &adapter_state.done)) return error.WgpuUnavailable;
        const adapter = adapter_state.adapter orelse return error.WgpuUnavailable;
        if (adapter_state.status != c.WGPURequestAdapterStatus_Success) return error.WgpuUnavailable;
        errdefer c.wgpuAdapterRelease(adapter);

        var device_state = DeviceRequest{};
        var device_cb: c.WGPURequestDeviceCallbackInfo = std.mem.zeroes(c.WGPURequestDeviceCallbackInfo);
        device_cb.mode = c.WGPUCallbackMode_AllowProcessEvents;
        device_cb.callback = requestDeviceCallback;
        device_cb.userdata1 = &device_state;
        var device_desc: c.WGPUDeviceDescriptor = std.mem.zeroes(c.WGPUDeviceDescriptor);
        _ = c.wgpuAdapterRequestDevice(adapter, &device_desc, device_cb);
        if (!waitForCallback(instance, &device_state.done)) return error.WgpuUnavailable;
        const device = device_state.device orelse return error.WgpuUnavailable;
        if (device_state.status != c.WGPURequestDeviceStatus_Success) return error.WgpuUnavailable;
        errdefer c.wgpuDeviceRelease(device);

        const queue = c.wgpuDeviceGetQueue(device) orelse return error.WgpuUnavailable;
        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
        };
    }

    fn retain(self: GpuContext) GpuContext {
        c.wgpuInstanceAddRef(self.instance);
        c.wgpuAdapterAddRef(self.adapter);
        c.wgpuDeviceAddRef(self.device);
        c.wgpuQueueAddRef(self.queue);
        return self;
    }

    fn deinit(self: *GpuContext) void {
        c.wgpuQueueRelease(self.queue);
        c.wgpuDeviceRelease(self.device);
        c.wgpuAdapterRelease(self.adapter);
        c.wgpuInstanceRelease(self.instance);
        self.* = undefined;
    }
};

const AdapterRequest = struct {
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    status: c.WGPURequestAdapterStatus = c.WGPURequestAdapterStatus_Error,
    adapter: c.WGPUAdapter = null,
};

const DeviceRequest = struct {
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    status: c.WGPURequestDeviceStatus = c.WGPURequestDeviceStatus_Error,
    device: c.WGPUDevice = null,
};

const MapRequest = struct {
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Error,
};

fn requestAdapterCallback(status: c.WGPURequestAdapterStatus, adapter: c.WGPUAdapter, message: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const state: *AdapterRequest = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.adapter = adapter;
    state.done.store(true, .release);
}

fn requestDeviceCallback(status: c.WGPURequestDeviceStatus, device: c.WGPUDevice, message: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const state: *DeviceRequest = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.device = device;
    state.done.store(true, .release);
}

fn mapCallback(status: c.WGPUMapAsyncStatus, message: c.WGPUStringView, userdata1: ?*anyopaque, userdata2: ?*anyopaque) callconv(.c) void {
    _ = message;
    _ = userdata2;
    const state: *MapRequest = @ptrCast(@alignCast(userdata1.?));
    state.status = status;
    state.done.store(true, .release);
}

fn waitForCallback(instance: c.WGPUInstance, done: *std.atomic.Value(bool)) bool {
    var remaining_polls: usize = wait_timeout_ns / 1_000_000;
    while (!done.load(.acquire) and remaining_polls > 0) : (remaining_polls -= 1) {
        c.wgpuInstanceProcessEvents(instance);
        sleepOneMillisecond();
    }
    return done.load(.acquire);
}

fn sleepOneMillisecond() void {
    var ts = std.c.timespec{
        .sec = 0,
        .nsec = 1_000_000,
    };
    _ = std.c.nanosleep(&ts, null);
}

const PipelineResources = struct {
    pipeline: c.WGPUComputePipeline,
    bind_group_layout: c.WGPUBindGroupLayout,
};

const CompiledDispatch = struct {
    shape: ProgramShape,
    op_start: usize,
    op_count: usize,
    pipeline: c.WGPUComputePipeline,
    bind_group_layout: c.WGPUBindGroupLayout,

    fn opSlice(self: CompiledDispatch, ops: []const backend_mod.DeviceOp) []const backend_mod.DeviceOp {
        return ops[self.op_start .. self.op_start + self.op_count];
    }
};

fn releaseDispatches(alloc: std.mem.Allocator, dispatches: []CompiledDispatch) void {
    for (dispatches) |dispatch| {
        c.wgpuBindGroupLayoutRelease(dispatch.bind_group_layout);
        c.wgpuComputePipelineRelease(dispatch.pipeline);
    }
    if (dispatches.len > 0) alloc.free(dispatches);
}

const CompiledProgram = struct {
    gpu: GpuContext,
    dispatches: []CompiledDispatch,
    program_stencil: program_mod.ProgramStencil,
    qweights: []DeviceQWeight,
    default_runtime: RuntimeBindings,
    buffer_sizes: []usize,
    initial_uploads: []backend_mod.ProgramIO,
    initial_upload_bytes: []u8,
    registered_resources: std.ArrayListUnmanaged(RegisteredResource),
    alloc: std.mem.Allocator,

    fn deinit(self: *CompiledProgram) void {
        self.default_runtime.deinit();
        if (self.initial_uploads.len > 0) self.alloc.free(self.initial_uploads);
        if (self.initial_upload_bytes.len > 0) self.alloc.free(self.initial_upload_bytes);
        releaseDeviceQWeights(self.alloc, self.qweights);
        for (self.registered_resources.items) |registered| {
            if (registered.release_on_drop) c.wgpuBufferRelease(registered.buffer);
        }
        self.registered_resources.deinit(self.alloc);
        releaseDispatches(self.alloc, self.dispatches);
        self.program_stencil.deinit(self.alloc);
        self.alloc.free(self.buffer_sizes);
        self.gpu.deinit();
        self.alloc.destroy(self);
    }

    fn outputBuffer(self: *CompiledProgram) u16 {
        std.debug.assert(self.dispatches.len > 0);
        return self.dispatches[self.dispatches.len - 1].shape.outputBuffer();
    }

    fn outputWriteCount(self: *CompiledProgram, buf_idx: u16) usize {
        var count: usize = 0;
        for (self.dispatches) |dispatch| {
            if (dispatch.shape.outputBuffer() == buf_idx) count += 1;
        }
        return count;
    }

    fn registerExternalBuffer(self: *CompiledProgram, buffer: c.WGPUBuffer, byte_len: u64) !backend_mod.ProgramIO.ExternalResource {
        const handle = bufferHandleValue(buffer) orelse return error.WgpuUnavailable;
        if (byte_len == 0 or byte_len > std.math.maxInt(u32)) return error.InvalidProgramIO;
        try self.registered_resources.append(self.alloc, .{
            .handle = handle,
            .buffer = buffer,
            .byte_len = byte_len,
        });
        return .{
            .placement = .webgpu,
            .handle = handle,
            .byte_len = @intCast(byte_len),
            .access = .read_write,
        };
    }

    fn resolveExternalBuffer(self: *CompiledProgram, resource: backend_mod.ProgramIO.ExternalResource) ?RegisteredResource {
        if (resource.placement != .webgpu or resource.handle == 0) return null;
        for (self.registered_resources.items) |registered| {
            if (registered.handle == resource.handle and resource.byte_len <= registered.byte_len) return registered;
        }
        return null;
    }

    fn unregisterExternalBuffer(self: *CompiledProgram, resource: backend_mod.ProgramIO.ExternalResource) ?RegisteredResource {
        for (self.registered_resources.items, 0..) |registered, i| {
            if (registered.handle != resource.handle) continue;
            const removed = registered;
            _ = self.registered_resources.swapRemove(i);
            return removed;
        }
        return null;
    }

    fn patchRuntimeWindow(self: *CompiledProgram, runtime: *RuntimeBindings, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
        const status = runtime.program_stencil.patchRuntimeWindow(window);
        if (status == .changed) {
            for (self.dispatches, 0..) |dispatch, i| {
                dispatch.shape.writeRuntimeParams(self.gpu.queue, runtime.params_buffers[i], dispatch.opSlice(runtime.program_stencil.ops));
            }
        }
        runtime.runtime_profile.recordRuntimePatch(status);
        return status;
    }

    fn upload(self: *CompiledProgram, runtime: *RuntimeBindings, inputs: []const backend_mod.ProgramIO) void {
        if (!runtime.program_stencil.ioValid(inputs, &.{})) return;
        for (inputs) |io| {
            if (io.resource != null) continue;
            const idx: usize = io.buf_idx;
            if (idx >= runtime.buffers.len) return;
            c.wgpuQueueWriteBuffer(self.gpu.queue, runtime.buffers[idx], io.offset, io.host_ptr, io.size);
        }
    }

    fn execute(self: *CompiledProgram, runtime: *RuntimeBindings, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO, download_outputs: bool) void {
        if (!runtime.program_stencil.ioValid(inputs, outputs)) return;
        self.upload(runtime, inputs);
        runtime.ensureBindGroups(self) catch return;
        const encoder = c.wgpuDeviceCreateCommandEncoder(self.gpu.device, null) orelse return;
        defer c.wgpuCommandEncoderRelease(encoder);
        if (self.dispatches.len == 0) runtime.copyResourceInputs(self, encoder, inputs) catch return;
        for (self.dispatches, 0..) |dispatch, i| {
            // Each dispatch gets its own WebGPU usage scope. Composite Programs
            // often write an intermediate buffer in one dispatch and read it in
            // the next, which is illegal inside a single compute pass.
            const pass = c.wgpuCommandEncoderBeginComputePass(encoder, null) orelse return;
            const bind_group = runtime.bind_groups[i].?;
            c.wgpuComputePassEncoderSetPipeline(pass, dispatch.pipeline);
            c.wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, null);
            c.wgpuComputePassEncoderDispatchWorkgroups(pass, dispatch.shape.dispatchWorkgroups(), 1, 1);
            c.wgpuComputePassEncoderEnd(pass);
            c.wgpuComputePassEncoderRelease(pass);
        }

        runtime.copyResourceOutputs(self, encoder, outputs) catch return;

        const should_download = download_outputs and outputsContainHost(outputs);
        if (should_download) {
            runtime.ensureReadbackCapacity(self, outputs) catch return;
            var readback_offset: u64 = 0;
            for (outputs) |io| {
                if (io.resource != null) continue;
                if (io.size == 0) continue;
                const idx: usize = io.buf_idx;
                if (idx >= runtime.buffers.len) return;
                const size: u64 = io.size;
                if (readback_offset > runtime.readback_byte_len or size > runtime.readback_byte_len - readback_offset) return;
                c.wgpuCommandEncoderCopyBufferToBuffer(
                    encoder,
                    runtime.buffers[idx],
                    io.offset,
                    runtime.readback_buffer,
                    readback_offset,
                    size,
                );
                readback_offset += size;
            }
        }

        const command = c.wgpuCommandEncoderFinish(encoder, null) orelse return;
        defer c.wgpuCommandBufferRelease(command);
        var commands = [_]c.WGPUCommandBuffer{command};
        c.wgpuQueueSubmit(self.gpu.queue, commands.len, &commands);

        if (should_download) runtime.download(self, outputs) catch return;
        runtime.runtime_profile.recordProgramCommandShapeDispatch(runtime.program_stencil.kernel_plan.command_shape);
        const actual_dispatches: u64 = @intCast(self.dispatches.len);
        const command_dispatches: u64 = @intCast(runtime.program_stencil.kernel_plan.command_shape.command_count);
        if (actual_dispatches >= command_dispatches) {
            runtime.runtime_profile.backend_dispatch_count +%= actual_dispatches - command_dispatches;
        } else {
            runtime.runtime_profile.backend_dispatch_count -= command_dispatches - actual_dispatches;
        }
        runtime.runtime_profile.backend_op_count +%= @intCast(runtime.program_stencil.ops.len);
        runtime.runtime_profile.call_count += 1;
    }
};

const RegisteredResource = struct {
    handle: usize,
    buffer: c.WGPUBuffer,
    byte_len: u64,
    release_on_drop: bool = false,
};

const DeviceQWeight = struct {
    data: c.WGPUBuffer,
    data_byte_len: u64,
    scales: c.WGPUBuffer,
    scales_byte_len: u64,
    rows: u32,
    cols: u32,
    block_size: u32,
};

fn releaseDeviceQWeights(alloc: std.mem.Allocator, qweights: []DeviceQWeight) void {
    for (qweights) |qw| {
        c.wgpuBufferRelease(qw.data);
        c.wgpuBufferRelease(qw.scales);
    }
    if (qweights.len > 0) alloc.free(qweights);
}

fn prepareDeviceQWeights(gpu: *GpuContext, alloc: std.mem.Allocator, qweights: []const backend_mod.QuantizedWeightUpload) ![]DeviceQWeight {
    if (qweights.len == 0) return &.{};
    const views = try alloc.alloc(DeviceQWeight, qweights.len);
    var made: usize = 0;
    errdefer {
        for (views[0..made]) |qw| {
            c.wgpuBufferRelease(qw.data);
            c.wgpuBufferRelease(qw.scales);
        }
        alloc.free(views);
    }

    for (qweights, 0..) |qw, i| {
        views[i] = try prepareDeviceQWeight(gpu, alloc, qw);
        made += 1;
    }
    return views;
}

fn prepareDeviceQWeight(gpu: *GpuContext, alloc: std.mem.Allocator, qw: backend_mod.QuantizedWeightUpload) !DeviceQWeight {
    const rows = u32Value(qw.rows) orelse return error.InvalidQuantizedWeight;
    const cols = u32Value(qw.cols) orelse return error.InvalidQuantizedWeight;
    const block_size = u32Value(qw.block_size) orelse return error.InvalidQuantizedWeight;
    if (rows == 0 or cols == 0 or block_size == 0) return error.InvalidQuantizedWeight;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return error.InvalidQuantizedWeight;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return error.InvalidQuantizedWeight;

    const packed_len = (n_data + 3) / 4;
    const packed_words = try alloc.alloc(u32, packed_len);
    defer alloc.free(packed_words);
    @memset(packed_words, 0);
    for (qw.data[0..n_data], 0..) |value, idx| {
        const byte: u8 = @bitCast(value);
        const word_idx = idx / 4;
        const shift: u5 = @intCast((idx % 4) * 8);
        packed_words[word_idx] |= @as(u32, byte) << shift;
    }

    const data_bytes = std.math.mul(usize, packed_words.len, @sizeOf(u32)) catch return error.InvalidQuantizedWeight;
    const scales_bytes = std.math.mul(usize, n_blocks, @sizeOf(f32)) catch return error.InvalidQuantizedWeight;
    const data_buffer = try createBuffer(gpu.device, @intCast(data_bytes), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    errdefer c.wgpuBufferRelease(data_buffer);
    const scales_buffer = try createBuffer(gpu.device, @intCast(scales_bytes), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    errdefer c.wgpuBufferRelease(scales_buffer);
    c.wgpuQueueWriteBuffer(gpu.queue, data_buffer, 0, packed_words.ptr, data_bytes);
    c.wgpuQueueWriteBuffer(gpu.queue, scales_buffer, 0, qw.scales.ptr, scales_bytes);

    return .{
        .data = data_buffer,
        .data_byte_len = @intCast(data_bytes),
        .scales = scales_buffer,
        .scales_byte_len = @intCast(scales_bytes),
        .rows = rows,
        .cols = cols,
        .block_size = block_size,
    };
}

const BindingBuffer = struct {
    buffer: c.WGPUBuffer,
    offset: u64,
    size: u64,
};

const RuntimeBindings = struct {
    buffers: []c.WGPUBuffer,
    params_buffers: []c.WGPUBuffer,
    readback_buffer: c.WGPUBuffer,
    readback_byte_len: u64,
    bind_groups: []?c.WGPUBindGroup,
    configured_persistent: []backend_mod.ProgramIO = &.{},
    configured_inputs: []backend_mod.ProgramIO = &.{},
    configured_outputs: []backend_mod.ProgramIO = &.{},
    qweights: []DeviceQWeight = &.{},
    program_stencil: program_mod.ProgramStencil,
    runtime_profile: profile_mod.RuntimeProfile,
    alloc: std.mem.Allocator,

    fn init(alloc: std.mem.Allocator, compiled: *CompiledProgram) !RuntimeBindings {
        const buffers = try alloc.alloc(c.WGPUBuffer, compiled.buffer_sizes.len);
        errdefer alloc.free(buffers);
        var made: usize = 0;
        errdefer {
            for (buffers[0..made]) |buffer| c.wgpuBufferRelease(buffer);
        }
        for (compiled.buffer_sizes, 0..) |elements, i| {
            const bytes = try bytesForElements(elements);
            buffers[i] = try createBuffer(compiled.gpu.device, bytes, c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc);
            made += 1;
        }

        const readback_byte_len = if (compiled.dispatches.len == 0)
            1
        else
            try bytesForElements(compiled.buffer_sizes[compiled.outputBuffer()]);
        const readback = try createBuffer(compiled.gpu.device, readback_byte_len, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        errdefer c.wgpuBufferRelease(readback);

        const params = try alloc.alloc(c.WGPUBuffer, compiled.dispatches.len);
        errdefer alloc.free(params);
        var params_made: usize = 0;
        errdefer {
            for (params[0..params_made]) |buffer| c.wgpuBufferRelease(buffer);
        }
        for (compiled.dispatches, 0..) |dispatch, i| {
            params[i] = try createBuffer(compiled.gpu.device, dispatch.shape.paramsByteLen(), c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst);
            params_made += 1;
        }

        const bind_groups = try alloc.alloc(?c.WGPUBindGroup, compiled.dispatches.len);
        errdefer alloc.free(bind_groups);
        @memset(bind_groups, null);

        var runtime_stencil = try compiled.program_stencil.clone(alloc);
        errdefer runtime_stencil.deinit(alloc);
        for (compiled.dispatches, 0..) |dispatch, i| {
            dispatch.shape.writeRuntimeParams(compiled.gpu.queue, params[i], dispatch.opSlice(runtime_stencil.ops));
        }
        const inspection = runtime_stencil.inspect();
        var runtime = RuntimeBindings{
            .buffers = buffers,
            .params_buffers = params,
            .readback_buffer = readback,
            .readback_byte_len = readback_byte_len,
            .bind_groups = bind_groups,
            .qweights = &.{},
            .program_stencil = runtime_stencil,
            .runtime_profile = .{
                .runtime_patch_shape = inspection.runtime_patch_shape,
                .program_command_shape = inspection.command_shape,
            },
            .alloc = alloc,
        };
        compiled.upload(&runtime, compiled.initial_uploads);
        return runtime;
    }

    fn configure(self: *RuntimeBindings, compiled: *CompiledProgram, persistent: []const backend_mod.ProgramIO, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) !void {
        if (!self.program_stencil.ioValid(persistent, &.{})) return error.InvalidProgramIO;
        if (!self.program_stencil.ioValid(inputs, outputs)) return error.InvalidProgramIO;
        try self.validateResources(compiled, persistent, .read);
        try self.validateResources(compiled, inputs, .read);
        try self.validateResources(compiled, outputs, .write);
        try self.ensureReadbackCapacity(compiled, outputs);
        const next_persistent = try self.alloc.dupe(backend_mod.ProgramIO, persistent);
        errdefer self.alloc.free(next_persistent);
        const next_inputs = try self.alloc.dupe(backend_mod.ProgramIO, inputs);
        errdefer self.alloc.free(next_inputs);
        const next_outputs = try self.alloc.dupe(backend_mod.ProgramIO, outputs);
        errdefer self.alloc.free(next_outputs);
        if (self.configured_persistent.len > 0) self.alloc.free(self.configured_persistent);
        if (self.configured_inputs.len > 0) self.alloc.free(self.configured_inputs);
        if (self.configured_outputs.len > 0) self.alloc.free(self.configured_outputs);
        self.configured_persistent = next_persistent;
        self.configured_inputs = next_inputs;
        self.configured_outputs = next_outputs;
        try self.rebuildBindGroups(compiled);
    }

    const RequiredResourceAccess = enum {
        read,
        write,
    };

    fn validateResources(self: *RuntimeBindings, compiled: *CompiledProgram, ios: []const backend_mod.ProgramIO, required_access: RequiredResourceAccess) !void {
        _ = self;
        for (ios) |io| {
            const resource = io.resource orelse continue;
            switch (required_access) {
                .read => if (!resource.canRead()) return error.UnsupportedResourceBinding,
                .write => if (!resource.canWrite()) return error.UnsupportedResourceBinding,
            }
            const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
            const begin: u64 = resource.byte_offset;
            if (begin % storage_buffer_binding_alignment != 0) return error.UnsupportedResourceBinding;
            if (begin > registered.byte_len or resource.byte_len > registered.byte_len - begin) return error.InvalidProgramIO;
            if (@as(u64, io.size) > resource.byte_len) return error.InvalidProgramIO;
        }
    }

    fn ensureReadbackCapacity(self: *RuntimeBindings, compiled: *CompiledProgram, outputs: []const backend_mod.ProgramIO) !void {
        const required = try readbackByteLenForOutputs(outputs);
        if (required == 0 or required <= self.readback_byte_len) return;
        const readback = try createBuffer(compiled.gpu.device, required, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
        c.wgpuBufferRelease(self.readback_buffer);
        self.readback_buffer = readback;
        self.readback_byte_len = required;
    }

    fn ensureBindGroups(self: *RuntimeBindings, compiled: *CompiledProgram) !void {
        for (self.bind_groups) |bind_group| {
            if (bind_group == null) {
                try self.rebuildBindGroups(compiled);
                return;
            }
        }
    }

    fn releaseBindGroups(self: *RuntimeBindings) void {
        for (self.bind_groups) |*bind_group| {
            if (bind_group.*) |group| {
                c.wgpuBindGroupRelease(group);
                bind_group.* = null;
            }
        }
    }

    fn rebuildBindGroups(self: *RuntimeBindings, compiled: *CompiledProgram) !void {
        self.releaseBindGroups();
        for (compiled.dispatches, 0..) |dispatch, i| {
            switch (dispatch.shape) {
                .tiny_linear => |shape| try self.rebuildTinyLinearBindGroup(compiled, i, shape),
                .matmul => |shape| try self.rebuildMatmulBindGroup(compiled, i, shape),
                .matmul_elementwise => |shape| try self.rebuildMatmulElementwiseBindGroup(compiled, i, shape),
                .matmul_fused_elementwise => |shape| try self.rebuildMatmulFusedElementwiseBindGroup(compiled, i, shape),
                .matvec_elementwise => |shape| try self.rebuildMatvecElementwiseBindGroup(compiled, i, shape),
                .matvec_fused_elementwise => |shape| try self.rebuildMatvecFusedElementwiseBindGroup(compiled, i, shape),
                .matvec_slice => |shape| try self.rebuildMatvecSliceBindGroup(compiled, i, shape),
                .matvec_rope_slice => |shape| try self.rebuildMatvecRopeSliceBindGroup(compiled, i, shape),
                .qmatmul => |shape| try self.rebuildQMatmulBindGroup(compiled, i, shape),
                .qmatmul_elementwise => |shape| try self.rebuildQMatmulElementwiseBindGroup(compiled, i, shape),
                .qmatvec_slice => |shape| try self.rebuildQMatvecSliceBindGroup(compiled, i, shape),
                .qmatvec_rope_slice => |shape| try self.rebuildQMatvecRopeSliceBindGroup(compiled, i, shape),
                .qmatvec_elementwise => |shape| try self.rebuildQMatvecElementwiseBindGroup(compiled, i, shape),
                .qmatvec_fused_elementwise => |shape| try self.rebuildQMatvecFusedElementwiseBindGroup(compiled, i, shape),
                .qlinear => |shape| try self.rebuildQLinearBindGroup(compiled, i, shape),
                .elementwise => |shape| try self.rebuildElementwiseBindGroup(compiled, i, shape),
                .fused_elementwise => |shape| try self.rebuildFusedElementwiseBindGroup(compiled, i, shape),
                .repeat => |shape| try self.rebuildRepeatBindGroup(compiled, i, shape),
                .layernorm => |shape| try self.rebuildLayerNormBindGroup(compiled, i, shape),
                .rmsnorm => |shape| try self.rebuildRmsNormBindGroup(compiled, i, shape),
                .softmax => |shape| try self.rebuildSoftmaxBindGroup(compiled, i, shape),
                .logsoftmax => |shape| try self.rebuildSoftmaxBindGroup(compiled, i, shape),
                .reduce => |shape| try self.rebuildReduceBindGroup(compiled, i, shape),
                .rope => |shape| try self.rebuildRopeBindGroup(compiled, i, shape),
                .slice_assign => |shape| try self.rebuildSliceAssignBindGroup(compiled, i, shape),
                .attention => |shape| try self.rebuildAttentionBindGroup(compiled, i, shape),
            }
        }
    }

    fn rebuildTinyLinearBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: TinyLinearShape) !void {
        const input = try self.bindingBufferFor(compiled, shape.input, self.configured_inputs);
        const weights = try self.bindingBufferFor(compiled, shape.weights, self.configured_persistent);
        const bias = try self.bindingBufferFor(compiled, shape.bias, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, weights),
            bindGroupBufferEntry(2, bias),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(LinearParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatmulBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatmulShape) !void {
        const a = try self.bindingBufferForEither(compiled, shape.a, self.configured_inputs, self.configured_persistent);
        const b = try self.bindingBufferForEither(compiled, shape.b, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, a),
            bindGroupBufferEntry(1, b),
            bindGroupBufferEntry(2, output),
            bindGroupBufferEntry(3, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(MatmulParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatmulElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatmulElementwiseShape) !void {
        const a = try self.bindingBufferForEither(compiled, shape.a, self.configured_inputs, self.configured_persistent);
        const b = try self.bindingBufferForEither(compiled, shape.b, self.configured_persistent, self.configured_inputs);
        const secondary = try self.bindingBufferForEither(compiled, shape.secondary, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, a),
            bindGroupBufferEntry(1, b),
            bindGroupBufferEntry(2, secondary),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(MatmulElementwiseParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatmulFusedElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatmulFusedElementwiseShape) !void {
        const a = try self.bindingBufferForEither(compiled, shape.a, self.configured_inputs, self.configured_persistent);
        const b = try self.bindingBufferForEither(compiled, shape.b, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries: [3 + max_matmul_fused_elementwise_secondaries + 1]c.WGPUBindGroupEntry = undefined;
        entries[0] = bindGroupBufferEntry(0, a);
        entries[1] = bindGroupBufferEntry(1, b);
        entries[2] = bindGroupBufferEntry(2, output);
        for (shape.secondary_bufs, 0..) |buf_idx, i| {
            const secondary = try self.bindingBufferForEither(compiled, buf_idx, self.configured_inputs, self.configured_persistent);
            entries[3 + i] = bindGroupBufferEntry(@intCast(3 + i), secondary);
        }
        entries[3 + max_matmul_fused_elementwise_secondaries] = bindGroupBufferEntry(3 + max_matmul_fused_elementwise_secondaries, .{
            .buffer = self.params_buffers[dispatch_idx],
            .offset = 0,
            .size = @sizeOf(MatmulFusedElementwiseParams),
        });
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatvecElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatvecElementwiseShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const weights = try self.bindingBufferForEither(compiled, shape.weights, self.configured_persistent, self.configured_inputs);
        const secondary = try self.bindingBufferForEither(compiled, shape.secondary, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, weights),
            bindGroupBufferEntry(2, secondary),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(MatvecElementwiseParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatvecFusedElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatvecFusedElementwiseShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const weights = try self.bindingBufferForEither(compiled, shape.weights, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries: [3 + max_matvec_fused_elementwise_secondaries + 1]c.WGPUBindGroupEntry = undefined;
        entries[0] = bindGroupBufferEntry(0, input);
        entries[1] = bindGroupBufferEntry(1, weights);
        entries[2] = bindGroupBufferEntry(2, output);
        for (shape.secondary_bufs, 0..) |buf_idx, i| {
            const secondary = try self.bindingBufferForEither(compiled, buf_idx, self.configured_inputs, self.configured_persistent);
            entries[3 + i] = bindGroupBufferEntry(@intCast(3 + i), secondary);
        }
        entries[3 + max_matvec_fused_elementwise_secondaries] = bindGroupBufferEntry(3 + max_matvec_fused_elementwise_secondaries, .{
            .buffer = self.params_buffers[dispatch_idx],
            .offset = 0,
            .size = @sizeOf(MatvecFusedElementwiseParams),
        });
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatvecSliceBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatvecSliceShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const weights = try self.bindingBufferForEither(compiled, shape.weights, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, weights),
            bindGroupBufferEntry(2, output),
            bindGroupBufferEntry(3, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(MatvecSliceParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildMatvecRopeSliceBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: MatvecRopeSliceShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const weights = try self.bindingBufferForEither(compiled, shape.weights, self.configured_persistent, self.configured_inputs);
        const cos_sin = try self.bindingBufferForEither(compiled, shape.cos_sin, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, weights),
            bindGroupBufferEntry(2, cos_sin),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(MatvecRopeSliceParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatmulBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatmulShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QMatmulParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatmulElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatmulElementwiseShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const secondary = try self.bindingBufferForEither(compiled, shape.secondary, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, secondary),
            bindGroupBufferEntry(4, output),
            bindGroupBufferEntry(5, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QMatmulElementwiseParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatvecSliceBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatvecSliceShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, output),
            bindGroupBufferEntry(4, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QMatvecSliceParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatvecRopeSliceBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatvecRopeSliceShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const cos_sin = try self.bindingBufferForEither(compiled, shape.cos_sin, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, cos_sin),
            bindGroupBufferEntry(4, output),
            bindGroupBufferEntry(5, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QMatvecRopeSliceParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatvecElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatvecElementwiseShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const secondary = try self.bindingBufferForEither(compiled, shape.secondary, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, secondary),
            bindGroupBufferEntry(4, output),
            bindGroupBufferEntry(5, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QMatvecElementwiseParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQMatvecFusedElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QMatvecFusedElementwiseShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries: [4 + max_qmatvec_fused_elementwise_secondaries + 1]c.WGPUBindGroupEntry = undefined;
        entries[0] = bindGroupBufferEntry(0, input);
        entries[1] = bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len });
        entries[2] = bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len });
        entries[3] = bindGroupBufferEntry(3, output);
        for (shape.secondary_bufs, 0..) |buf_idx, i| {
            const secondary = try self.bindingBufferForEither(compiled, buf_idx, self.configured_inputs, self.configured_persistent);
            entries[4 + i] = bindGroupBufferEntry(@intCast(4 + i), secondary);
        }
        entries[4 + max_qmatvec_fused_elementwise_secondaries] = bindGroupBufferEntry(4 + max_qmatvec_fused_elementwise_secondaries, .{
            .buffer = self.params_buffers[dispatch_idx],
            .offset = 0,
            .size = @sizeOf(QMatvecFusedElementwiseParams),
        });
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildQLinearBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: QLinearShape) !void {
        const input = try self.bindingBufferForEither(compiled, shape.input, self.configured_inputs, self.configured_persistent);
        const qweight_view = try self.qweight(compiled, shape.qweight_idx);
        const bias = try self.bindingBufferForEither(compiled, shape.bias, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, input),
            bindGroupBufferEntry(1, .{ .buffer = qweight_view.data, .offset = 0, .size = qweight_view.data_byte_len }),
            bindGroupBufferEntry(2, .{ .buffer = qweight_view.scales, .offset = 0, .size = qweight_view.scales_byte_len }),
            bindGroupBufferEntry(3, bias),
            bindGroupBufferEntry(4, output),
            bindGroupBufferEntry(5, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(QLinearParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn qweight(self: *RuntimeBindings, compiled: *CompiledProgram, idx: u16) !DeviceQWeight {
        const runtime_qweights = if (self.qweights.len > 0) self.qweights else compiled.qweights;
        const qidx: usize = idx;
        if (qidx >= runtime_qweights.len) return error.UnsupportedDeviceOp;
        return runtime_qweights[qidx];
    }

    fn uploadQWeights(self: *RuntimeBindings, compiled: *CompiledProgram, qweights: []const backend_mod.QuantizedWeightUpload) !void {
        if (qweights.len != compiled.qweights.len) return error.ShapeMismatch;
        for (qweights, compiled.qweights) |qw, expected| {
            if (qw.rows != expected.rows or qw.cols != expected.cols or qw.block_size != expected.block_size) return error.ShapeMismatch;
        }
        const prepared = try prepareDeviceQWeights(&compiled.gpu, self.alloc, qweights);
        errdefer releaseDeviceQWeights(self.alloc, prepared);
        releaseDeviceQWeights(self.alloc, self.qweights);
        self.qweights = prepared;
        self.releaseBindGroups();
    }

    fn rebuildElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: ElementwiseShape) !void {
        const src0 = try self.bindingBufferForEither(compiled, shape.src0, self.configured_inputs, self.configured_persistent);
        const src1 = try self.bindingBufferForEither(compiled, shape.src1, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src0),
            bindGroupBufferEntry(1, src1),
            bindGroupBufferEntry(2, output),
            bindGroupBufferEntry(3, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(ElementwiseParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildFusedElementwiseBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: FusedElementwiseShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries: [2 + max_fused_elementwise_secondaries + 1]c.WGPUBindGroupEntry = undefined;
        entries[0] = bindGroupBufferEntry(0, src);
        entries[1] = bindGroupBufferEntry(1, output);
        for (shape.secondary_bufs, 0..) |buf_idx, i| {
            const secondary = try self.bindingBufferForEither(compiled, buf_idx, self.configured_inputs, self.configured_persistent);
            entries[2 + i] = bindGroupBufferEntry(@intCast(2 + i), secondary);
        }
        entries[2 + max_fused_elementwise_secondaries] = bindGroupBufferEntry(2 + max_fused_elementwise_secondaries, .{
            .buffer = self.params_buffers[dispatch_idx],
            .offset = 0,
            .size = @sizeOf(FusedElementwiseParams),
        });
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildRepeatBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: RepeatShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(RepeatParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildLayerNormBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: LayerNormShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(LayerNormParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildRmsNormBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: RmsNormShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(RmsNormParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildSoftmaxBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: SoftmaxShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(SoftmaxParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildReduceBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: ReduceShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(ReduceParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildRopeBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: RopeShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const cos_sin = try self.bindingBufferForEither(compiled, shape.cos_sin, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, cos_sin),
            bindGroupBufferEntry(2, output),
            bindGroupBufferEntry(3, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(RopeParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildSliceAssignBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: SliceAssignShape) !void {
        const src = try self.bindingBufferForEither(compiled, shape.src, self.configured_inputs, self.configured_persistent);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, src),
            bindGroupBufferEntry(1, output),
            bindGroupBufferEntry(2, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(SliceAssignParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn rebuildAttentionBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, shape: AttentionShape) !void {
        const q = try self.bindingBufferForEither(compiled, shape.q, self.configured_inputs, self.configured_persistent);
        const k = try self.bindingBufferForEither(compiled, shape.k, self.configured_persistent, self.configured_inputs);
        const v = try self.bindingBufferForEither(compiled, shape.v, self.configured_persistent, self.configured_inputs);
        const mask = try self.bindingBufferForEither(compiled, shape.mask, self.configured_persistent, self.configured_inputs);
        const output = try self.outputBindingBufferFor(compiled, shape.output);
        var entries = [_]c.WGPUBindGroupEntry{
            bindGroupBufferEntry(0, q),
            bindGroupBufferEntry(1, k),
            bindGroupBufferEntry(2, v),
            bindGroupBufferEntry(3, mask),
            bindGroupBufferEntry(4, output),
            bindGroupBufferEntry(5, .{ .buffer = self.params_buffers[dispatch_idx], .offset = 0, .size = @sizeOf(AttentionParams) }),
        };
        try self.createBindGroup(compiled, dispatch_idx, &entries);
    }

    fn createBindGroup(self: *RuntimeBindings, compiled: *CompiledProgram, dispatch_idx: usize, entries: []const c.WGPUBindGroupEntry) !void {
        var desc: c.WGPUBindGroupDescriptor = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        desc.layout = compiled.dispatches[dispatch_idx].bind_group_layout;
        desc.entryCount = entries.len;
        desc.entries = entries.ptr;
        self.bind_groups[dispatch_idx] = c.wgpuDeviceCreateBindGroup(compiled.gpu.device, &desc) orelse return error.WgpuUnavailable;
    }

    fn bindingBufferForEither(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16, preferred: []const backend_mod.ProgramIO, fallback: []const backend_mod.ProgramIO) !BindingBuffer {
        if (try self.findBindingBufferFor(compiled, buf_idx, preferred)) |buffer| return buffer;
        if (try self.findBindingBufferFor(compiled, buf_idx, fallback)) |buffer| return buffer;
        return self.defaultBindingBufferFor(compiled, buf_idx);
    }

    fn outputBindingBufferFor(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16) !BindingBuffer {
        if (try self.findOutputBindingBufferFor(compiled, buf_idx)) |buffer| return buffer;
        if (try self.findBindingBufferFor(compiled, buf_idx, self.configured_persistent)) |buffer| return buffer;
        return self.defaultBindingBufferFor(compiled, buf_idx);
    }

    fn bindingBufferFor(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16, ios: []const backend_mod.ProgramIO) !BindingBuffer {
        if (try self.findBindingBufferFor(compiled, buf_idx, ios)) |buffer| return buffer;
        return self.defaultBindingBufferFor(compiled, buf_idx);
    }

    fn outputResourceUsesExternalStorage(self: *RuntimeBindings, compiled: *CompiledProgram, io: backend_mod.ProgramIO) bool {
        if (io.resource == null) return false;
        if (self.findIOFor(io.buf_idx, self.configured_persistent)) |persistent| {
            if (persistent.resource != null) return true;
        }
        const write_count = compiled.outputWriteCount(io.buf_idx);
        if (write_count == 0) return false;
        return write_count <= 1;
    }

    fn findOutputBindingBufferFor(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16) !?BindingBuffer {
        const io = self.findIOFor(buf_idx, self.configured_outputs) orelse return null;
        if (io.resource != null and !self.outputResourceUsesExternalStorage(compiled, io)) {
            return try self.defaultBindingBufferFor(compiled, buf_idx);
        }
        return try self.bindingBufferFor(compiled, buf_idx, self.configured_outputs);
    }

    fn findIOFor(self: *RuntimeBindings, buf_idx: u16, ios: []const backend_mod.ProgramIO) ?backend_mod.ProgramIO {
        _ = self;
        for (ios) |io| {
            if (io.buf_idx == buf_idx) return io;
        }
        return null;
    }

    fn findBindingBufferFor(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16, ios: []const backend_mod.ProgramIO) !?BindingBuffer {
        for (ios) |io| {
            if (io.buf_idx != buf_idx) continue;
            if (io.resource) |resource| {
                if (io.offset != 0) return error.UnsupportedResourceBinding;
                const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
                return .{
                    .buffer = registered.buffer,
                    .offset = resource.byte_offset,
                    .size = io.size,
                };
            }
            return try self.defaultBindingBufferFor(compiled, buf_idx);
        }
        return null;
    }

    fn defaultBindingBufferFor(self: *RuntimeBindings, compiled: *CompiledProgram, buf_idx: u16) !BindingBuffer {
        return .{
            .buffer = self.buffers[buf_idx],
            .offset = 0,
            .size = try bytesForElements(compiled.buffer_sizes[buf_idx]),
        };
    }

    fn download(self: *RuntimeBindings, compiled: *CompiledProgram, outputs: []const backend_mod.ProgramIO) !void {
        const readback_byte_len = try readbackByteLenForOutputs(outputs);
        if (readback_byte_len == 0) return;
        if (readback_byte_len > self.readback_byte_len) return error.InvalidProgramIO;
        var map_state = MapRequest{};
        var map_cb: c.WGPUBufferMapCallbackInfo = std.mem.zeroes(c.WGPUBufferMapCallbackInfo);
        map_cb.mode = c.WGPUCallbackMode_AllowProcessEvents;
        map_cb.callback = mapCallback;
        map_cb.userdata1 = &map_state;
        _ = c.wgpuBufferMapAsync(self.readback_buffer, c.WGPUMapMode_Read, 0, readback_byte_len, map_cb);
        if (!waitForCallback(compiled.gpu.instance, &map_state.done)) return error.WgpuUnavailable;
        if (map_state.status != c.WGPUMapAsyncStatus_Success) return error.WgpuUnavailable;
        defer c.wgpuBufferUnmap(self.readback_buffer);
        const mapped = c.wgpuBufferGetMappedRange(self.readback_buffer, 0, readback_byte_len) orelse return error.WgpuUnavailable;
        const bytes = @as([*]const u8, @ptrCast(mapped))[0..@intCast(readback_byte_len)];
        var readback_offset: usize = 0;
        for (outputs) |io| {
            if (io.resource != null) continue;
            const len: usize = io.size;
            if (readback_offset > bytes.len or len > bytes.len - readback_offset) return error.InvalidProgramIO;
            @memcpy(io.host_ptr[0..len], bytes[readback_offset .. readback_offset + len]);
            readback_offset += len;
        }
        self.runtime_profile.sync_count +%= 1;
    }

    fn copyResourceInputs(self: *RuntimeBindings, compiled: *CompiledProgram, encoder: c.WGPUCommandEncoder, inputs: []const backend_mod.ProgramIO) !void {
        for (inputs) |io| {
            const resource = io.resource orelse continue;
            if (io.size == 0) continue;
            const idx: usize = io.buf_idx;
            if (idx >= self.buffers.len) return error.InvalidProgramIO;
            const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
            const internal_size = try bytesForElements(compiled.buffer_sizes[idx]);
            const src_offset: u64 = resource.byte_offset;
            const dst_offset: u64 = io.offset;
            const size: u64 = io.size;
            if (src_offset > registered.byte_len or size > registered.byte_len - src_offset) return error.InvalidProgramIO;
            if (dst_offset > internal_size or size > internal_size - dst_offset) return error.InvalidProgramIO;
            c.wgpuCommandEncoderCopyBufferToBuffer(
                encoder,
                registered.buffer,
                src_offset,
                self.buffers[idx],
                dst_offset,
                size,
            );
        }
    }

    fn copyResourceOutputs(self: *RuntimeBindings, compiled: *CompiledProgram, encoder: c.WGPUCommandEncoder, outputs: []const backend_mod.ProgramIO) !void {
        for (outputs) |io| {
            const resource = io.resource orelse continue;
            if (self.outputResourceUsesExternalStorage(compiled, io)) continue;
            if (io.size == 0) continue;
            const idx: usize = io.buf_idx;
            if (idx >= self.buffers.len) return error.InvalidProgramIO;
            const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
            const internal_size = try bytesForElements(compiled.buffer_sizes[idx]);
            const src_offset: u64 = io.offset;
            const dst_offset: u64 = resource.byte_offset;
            const size: u64 = io.size;
            if (src_offset > internal_size or size > internal_size - src_offset) return error.InvalidProgramIO;
            if (dst_offset > registered.byte_len or size > registered.byte_len - dst_offset) return error.InvalidProgramIO;
            c.wgpuCommandEncoderCopyBufferToBuffer(
                encoder,
                self.buffers[idx],
                src_offset,
                registered.buffer,
                dst_offset,
                size,
            );
        }
    }

    fn deinit(self: *RuntimeBindings) void {
        self.releaseBindGroups();
        self.alloc.free(self.bind_groups);
        for (self.buffers) |buffer| c.wgpuBufferRelease(buffer);
        self.alloc.free(self.buffers);
        for (self.params_buffers) |buffer| c.wgpuBufferRelease(buffer);
        self.alloc.free(self.params_buffers);
        c.wgpuBufferRelease(self.readback_buffer);
        if (self.configured_persistent.len > 0) self.alloc.free(self.configured_persistent);
        if (self.configured_inputs.len > 0) self.alloc.free(self.configured_inputs);
        if (self.configured_outputs.len > 0) self.alloc.free(self.configured_outputs);
        releaseDeviceQWeights(self.alloc, self.qweights);
        self.program_stencil.deinit(self.alloc);
        self.* = undefined;
    }
};

const LinearParams = extern struct {
    k: u32,
    n: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
};

const MatmulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    dst_row_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
};

const MatmulElementwiseParams = extern struct {
    op: u32,
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const MatmulFusedElementwiseParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    dst_row_stride: u32,
    a_offset: u32,
    b_offset: u32,
    primary_dst_offset: u32,
    ew_dst_offset: u32,
    n_steps: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const MatvecElementwiseParams = extern struct {
    op: u32,
    n: u32,
    k: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
    pad0: u32 = 0,
};

const MatvecFusedElementwiseParams = extern struct {
    n: u32,
    k: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    dst_offset: u32,
    n_steps: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const MatvecSliceParams = extern struct {
    n: u32,
    k: u32,
    cells: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    slice_src_col_start: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
    slice_rows: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const MatvecRopeSliceParams = extern struct {
    n: u32,
    k: u32,
    half_d: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    a_offset: u32,
    b_offset: u32,
    rope_src_col_start: u32,
    cs_off: u32,
    cs_cs: u32,
    rope_dst_offset: u32,
    rope_dst_row_stride: u32,
    rope_dst_col_stride: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
};

const QMatmulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    dst_offset: u32,
    dst_row_stride: u32,
};

const QMatmulElementwiseParams = extern struct {
    op: u32,
    m: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    input_row_stride: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
};

const QMatvecSliceParams = extern struct {
    n: u32,
    k: u32,
    block_size: u32,
    cells: u32,
    input_offset: u32,
    slice_src_col_start: u32,
    slice_dst_offset: u32,
    slice_dst_row_stride: u32,
    slice_dst_col_stride: u32,
    slice_rows: u32,
};

const QMatvecRopeSliceParams = extern struct {
    n: u32,
    k: u32,
    block_size: u32,
    half_d: u32,
    input_offset: u32,
    rope_src_col_start: u32,
    cs_off: u32,
    cs_cs: u32,
    rope_dst_offset: u32,
    rope_dst_row_stride: u32,
    rope_dst_col_stride: u32,
    pad0: u32 = 0,
};

const QMatvecElementwiseParams = extern struct {
    op: u32,
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    secondary_offset: u32,
    dst_offset: u32,
    is_swapped: u32,
};

const QMatvecFusedElementwiseParams = extern struct {
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    dst_offset: u32,
    n_steps: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_primary: [max_fused_elementwise_steps]u32,
};

const QLinearParams = extern struct {
    n: u32,
    k: u32,
    block_size: u32,
    input_offset: u32,
    bias_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
};

const ElementwiseParams = extern struct {
    op: u32,
    n: u32,
    dst_offset: u32,
    src0_offset: u32,
    src1_offset: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const FusedElementwiseParams = extern struct {
    n: u32,
    n_steps: u32,
    dst_offset: u32,
    src_offset: u32,
    op: [max_fused_elementwise_steps]u32,
    is_swapped: [max_fused_elementwise_steps]u32,
    secondary_slot: [max_fused_elementwise_steps]u32,
    secondary_offset: [max_fused_elementwise_steps]u32,
    secondary_is_repeat: [max_fused_elementwise_steps]u32,
    secondary_repeat_dst_offset: [max_fused_elementwise_steps]u32,
    repeat_src_offset: u32,
    repeat_src_ne: [4]u32,
    repeat_src_strides: [4]u32,
    repeat_dst_strides: [4]u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const RepeatParams = extern struct {
    n: u32,
    src_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
    src_ne: [4]u32,
    src_strides: [4]u32,
    dst_strides: [4]u32,
};

const LayerNormParams = extern struct {
    rows: u32,
    cols: u32,
    eps: f32,
    src_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const RmsNormParams = extern struct {
    rows: u32,
    cols: u32,
    eps: f32,
    src_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const SoftmaxParams = extern struct {
    rows: u32,
    cols: u32,
    src_offset: u32,
    dst_offset: u32,
};

const ReduceParams = extern struct {
    op: u32,
    n_out: u32,
    reduce_size: u32,
    src_offset: u32,
    dst_offset: u32,
    pad0: u32 = 0,
    pad1: u32 = 0,
    pad2: u32 = 0,
};

const RopeParams = extern struct {
    half_d: u32,
    seq_len: u32,
    src_off: u32,
    cs_off: u32,
    dst_off: u32,
    src_rs: u32,
    src_cs: u32,
    cs_cs: u32,
};

const SliceAssignParams = extern struct {
    rows: u32,
    cols: u32,
    dst_offset: u32,
    dst_row_stride: u32,
    dst_col_stride: u32,
    src_offset: u32,
    src_row_stride: u32,
    src_col_stride: u32,
};

const AttentionParams = extern struct {
    d_head: u32,
    seq_q: u32,
    seq_kv: u32,
    has_mask: u32,
    scale: f32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    mask_off: u32,
    dst_off: u32,
    q_rs: u32,
    q_cs: u32,
    k_rs: u32,
    k_cs: u32,
    v_rs: u32,
    v_cs: u32,
    mask_rs: u32,
    mask_cs: u32,
    dst_rs: u32,
    dst_cs: u32,
};

fn createBuffer(device: c.WGPUDevice, size: u64, usage: c.WGPUBufferUsage) !c.WGPUBuffer {
    var desc: c.WGPUBufferDescriptor = std.mem.zeroes(c.WGPUBufferDescriptor);
    desc.usage = usage;
    desc.size = size;
    return c.wgpuDeviceCreateBuffer(device, &desc) orelse error.WgpuUnavailable;
}

fn bufferHandleValue(buffer: c.WGPUBuffer) ?usize {
    const ptr = buffer orelse return null;
    return @intFromPtr(ptr);
}

fn deviceHandleValue(device: c.WGPUDevice) ?usize {
    const ptr = device orelse return null;
    return @intFromPtr(ptr);
}

fn bytesForElements(elements: usize) !u64 {
    const bytes = try std.math.mul(usize, elements, @sizeOf(f32));
    return std.math.cast(u64, bytes) orelse error.InvalidProgramIO;
}

fn bindGroupBufferEntry(binding: u32, binding_buffer: BindingBuffer) c.WGPUBindGroupEntry {
    var entry: c.WGPUBindGroupEntry = std.mem.zeroes(c.WGPUBindGroupEntry);
    entry.binding = binding;
    entry.buffer = binding_buffer.buffer;
    entry.offset = binding_buffer.offset;
    entry.size = binding_buffer.size;
    return entry;
}

fn outputsContainHost(outputs: []const backend_mod.ProgramIO) bool {
    for (outputs) |io| if (io.resource == null) return true;
    return false;
}

fn readbackByteLenForOutputs(outputs: []const backend_mod.ProgramIO) !u64 {
    var total: u64 = 0;
    for (outputs) |io| {
        if (io.resource != null) continue;
        total = try std.math.add(u64, total, io.size);
    }
    return total;
}

fn writeBufferF32(gpu: *GpuContext, buffer: c.WGPUBuffer, values: []const f32) !void {
    const bytes = try bytesForElements(values.len);
    c.wgpuQueueWriteBuffer(gpu.queue, buffer, 0, values.ptr, bytes);
}

fn readBufferF32(gpu: *GpuContext, buffer: c.WGPUBuffer, values: []f32) !void {
    try readBufferBytes(gpu, buffer, 0, std.mem.sliceAsBytes(values));
}

fn readBufferBytes(gpu: *GpuContext, buffer: c.WGPUBuffer, byte_offset: u64, dst: []u8) !void {
    const bytes: u64 = @intCast(dst.len);
    const readback = try createBuffer(gpu.device, bytes, c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(readback);

    const encoder = c.wgpuDeviceCreateCommandEncoder(gpu.device, null) orelse return error.WgpuUnavailable;
    defer c.wgpuCommandEncoderRelease(encoder);
    c.wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer, byte_offset, readback, 0, bytes);
    const command = c.wgpuCommandEncoderFinish(encoder, null) orelse return error.WgpuUnavailable;
    defer c.wgpuCommandBufferRelease(command);
    var commands = [_]c.WGPUCommandBuffer{command};
    c.wgpuQueueSubmit(gpu.queue, commands.len, &commands);

    var map_state = MapRequest{};
    var map_cb: c.WGPUBufferMapCallbackInfo = std.mem.zeroes(c.WGPUBufferMapCallbackInfo);
    map_cb.mode = c.WGPUCallbackMode_AllowProcessEvents;
    map_cb.callback = mapCallback;
    map_cb.userdata1 = &map_state;
    _ = c.wgpuBufferMapAsync(readback, c.WGPUMapMode_Read, 0, bytes, map_cb);
    if (!waitForCallback(gpu.instance, &map_state.done)) return error.WgpuUnavailable;
    if (map_state.status != c.WGPUMapAsyncStatus_Success) return error.WgpuUnavailable;
    defer c.wgpuBufferUnmap(readback);
    const mapped = c.wgpuBufferGetMappedRange(readback, 0, bytes) orelse return error.WgpuUnavailable;
    const src = @as([*]const u8, @ptrCast(mapped))[0..@intCast(bytes)];
    @memcpy(dst, src);
}

fn stringView(value: []const u8) c.WGPUStringView {
    return .{ .data = value.ptr, .length = value.len };
}

fn u32Value(value: usize) ?u32 {
    return std.math.cast(u32, value);
}

fn detectExecutableShape(program: backend_mod.DeviceProgram) ?ProgramShape {
    if (detectTinyLinear(program)) |shape| return .{ .tiny_linear = shape };
    if (detectMatmul(program)) |shape| return .{ .matmul = shape };
    if (detectMatvecElementwise(program)) |shape| return .{ .matvec_elementwise = shape };
    if (detectMatmulElementwise(program)) |shape| return .{ .matmul_elementwise = shape };
    if (detectMatvecFusedElementwise(program)) |shape| return .{ .matvec_fused_elementwise = shape };
    if (detectMatmulFusedElementwise(program)) |shape| return .{ .matmul_fused_elementwise = shape };
    if (detectMatvecSlice(program)) |shape| return .{ .matvec_slice = shape };
    if (detectMatvecRopeSlice(program)) |shape| return .{ .matvec_rope_slice = shape };
    if (detectQMatmul(program)) |shape| return .{ .qmatmul = shape };
    if (detectQMatvecSlice(program)) |shape| return .{ .qmatvec_slice = shape };
    if (detectQMatvecRopeSlice(program)) |shape| return .{ .qmatvec_rope_slice = shape };
    if (detectQLinear(program)) |shape| return .{ .qlinear = shape };
    if (detectQMatmulElementwise(program)) |shape| return .{ .qmatmul_elementwise = shape };
    if (detectQMatvecElementwise(program)) |shape| return .{ .qmatvec_elementwise = shape };
    if (detectQMatvecFusedElementwise(program)) |shape| return .{ .qmatvec_fused_elementwise = shape };
    if (detectElementwise(program)) |shape| return .{ .elementwise = shape };
    if (detectRepeat(program)) |shape| return .{ .repeat = shape };
    if (detectRepeatFusedElementwise(program)) |shape| return .{ .fused_elementwise = shape };
    if (detectFusedElementwise(program)) |shape| return .{ .fused_elementwise = shape };
    if (detectLayerNorm(program)) |shape| return .{ .layernorm = shape };
    if (detectRmsNorm(program)) |shape| return .{ .rmsnorm = shape };
    if (detectSoftmax(program)) |shape| return .{ .softmax = shape };
    if (detectLogSoftmax(program)) |shape| return .{ .logsoftmax = shape };
    if (detectReduce(program)) |shape| return .{ .reduce = shape };
    if (detectRope(program)) |shape| return .{ .rope = shape };
    if (detectSliceAssign(program)) |shape| return .{ .slice_assign = shape };
    if (detectAttention(program)) |shape| return .{ .attention = shape };
    return null;
}

const DetectedDispatch = struct {
    shape: ProgramShape,
    op_count: usize,
};

fn subProgram(program: backend_mod.DeviceProgram, start: usize, count: usize) backend_mod.DeviceProgram {
    return .{
        .ops = program.ops[start .. start + count],
        .n_buffers = program.n_buffers,
        .buffer_sizes = program.buffer_sizes,
        .initial_uploads = &.{},
        .qweights = program.qweights,
    };
}

fn detectDispatchAt(program: backend_mod.DeviceProgram, start: usize) ?DetectedDispatch {
    const remaining = program.ops.len - start;
    var count = @min(max_dispatch_ops, remaining);
    while (count > 0) : (count -= 1) {
        if (detectExecutableShape(subProgram(program, start, count))) |shape| {
            const detected = DetectedDispatch{ .shape = shape, .op_count = count };
            if (detectedDispatchElidesLivePrimary(program, start, detected)) continue;
            return detected;
        }
    }
    return null;
}

fn detectedDispatchElidesLivePrimary(program: backend_mod.DeviceProgram, start: usize, detected: DetectedDispatch) bool {
    return switch (detected.shape) {
        .tiny_linear,
        .matmul_elementwise,
        .matvec_elementwise,
        .matvec_fused_elementwise,
        .matmul_fused_elementwise,
        .matvec_slice,
        => program_mod.matmulPrimaryOutputHasExternalUsers(program.ops, start, start + 1),
        .matvec_rope_slice => blk: {
            const sidecars = [_]?usize{ start + 1, start + 2 };
            break :blk program_mod.matmulPrimaryOutputHasExternalUsersExcept(program.ops, start, &sidecars);
        },
        .qmatvec_elementwise,
        .qmatvec_fused_elementwise,
        .qmatvec_slice,
        .qlinear,
        .qmatmul_elementwise,
        => program_mod.projectionPrimaryOutputHasExternalUsers(program.ops, start, start + 1),
        .qmatvec_rope_slice => blk: {
            const sidecars = [_]?usize{ start + 1, start + 2 };
            break :blk program_mod.projectionPrimaryOutputHasExternalUsersExcept(program.ops, start, &sidecars);
        },
        else => false,
    };
}

fn countDispatches(program: backend_mod.DeviceProgram) ?usize {
    if (program.ops.len == 0) return 0;
    var count: usize = 0;
    var op_index: usize = 0;
    while (op_index < program.ops.len) {
        const dispatch = detectDispatchAt(program, op_index) orelse return null;
        count += 1;
        op_index += dispatch.op_count;
    }
    return count;
}

fn supportsDispatchPlan(program: backend_mod.DeviceProgram) bool {
    return countDispatches(program) != null;
}

pub const DispatchFamilyCounts = backend_mod.ExecutionFamilyCounts;
pub const DispatchPlanInspection = backend_mod.ExecutionPlanInspection;

fn initDispatchPlan(total_op_count: usize) DispatchPlanInspection {
    return .{
        .total_op_count = @intCast(total_op_count),
        .max_dispatch_ops_per_dispatch = max_dispatch_ops,
    };
}

fn addDispatchShape(families: *DispatchFamilyCounts, shape: ProgramShape) void {
    switch (shape) {
        .tiny_linear => families.tiny_linear += 1,
        .matmul => families.matmul += 1,
        .matmul_elementwise => families.matmul_elementwise += 1,
        .matmul_fused_elementwise => families.matmul_fused_elementwise += 1,
        .matvec_elementwise => families.matvec_elementwise += 1,
        .matvec_fused_elementwise => families.matvec_fused_elementwise += 1,
        .matvec_slice => families.matvec_slice += 1,
        .matvec_rope_slice => families.matvec_rope_slice += 1,
        .qmatmul => families.qmatmul += 1,
        .qmatmul_elementwise => families.qmatmul_elementwise += 1,
        .qmatvec_slice => families.qmatvec_slice += 1,
        .qmatvec_rope_slice => families.qmatvec_rope_slice += 1,
        .qmatvec_elementwise => families.qmatvec_elementwise += 1,
        .qmatvec_fused_elementwise => families.qmatvec_fused_elementwise += 1,
        .qlinear => families.qlinear += 1,
        .elementwise => families.elementwise += 1,
        .fused_elementwise => families.fused_elementwise += 1,
        .repeat => families.repeat += 1,
        .layernorm => families.layernorm += 1,
        .rmsnorm => families.rmsnorm += 1,
        .softmax => families.softmax += 1,
        .logsoftmax => families.softmax += 1,
        .reduce => families.reduce += 1,
        .rope => families.rope += 1,
        .slice_assign => families.slice_assign += 1,
        .attention => families.attention += 1,
    }
}

fn recordDispatchPlan(plan: *DispatchPlanInspection, shape: ProgramShape, op_count: usize) void {
    plan.dispatch_count += 1;
    plan.covered_op_count += @intCast(op_count);
    addDispatchShape(&plan.family_counts, shape);
}

pub fn inspectDispatchPlan(program: backend_mod.DeviceProgram) DispatchPlanInspection {
    var out = initDispatchPlan(program.ops.len);
    var op_index: usize = 0;
    while (op_index < program.ops.len) {
        const detected = detectDispatchAt(program, op_index) orelse {
            out.first_unsupported_op = @intCast(op_index);
            return out;
        };
        recordDispatchPlan(&out, detected.shape, detected.op_count);
        op_index += detected.op_count;
    }
    out.supported = true;
    return out;
}

pub fn inspectCompiledDispatchPlan(handle: backend_mod.Backend.CompiledHandle) DispatchPlanInspection {
    if (!options.use_wgpu) return initDispatchPlan(0);
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    var out = initDispatchPlan(compiled.program_stencil.ops.len);
    for (compiled.dispatches) |dispatch| {
        recordDispatchPlan(&out, dispatch.shape, dispatch.op_count);
    }
    if (out.covered_op_count == out.total_op_count) {
        out.supported = true;
    } else if (out.covered_op_count < out.total_op_count) {
        out.first_unsupported_op = out.covered_op_count;
    }
    return out;
}

fn detectTinyLinear(program: backend_mod.DeviceProgram) ?TinyLinearShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const add = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (add.op != .add) return null;
    if (mat.geom.M != 1 or mat.geom.K == 0 or mat.geom.N == 0) return null;
    if (mat.geom.a_offset != 0 or mat.geom.b_offset != 0 or mat.geom.dst_offset != 0) return null;
    if (mat.geom.a_row_stride != mat.geom.K or mat.geom.a_col_stride != 1) return null;
    if (mat.geom.b_row_stride != mat.geom.N or mat.geom.b_col_stride != 1) return null;
    if (mat.geom.dst_row_stride != mat.geom.N) return null;
    if (add.src0 != mat.dst or add.n != mat.geom.N) return null;
    if (add.dst_offset != 0 or add.src0_offset != 0 or add.src1_offset != 0) return null;
    if (add.src1 == add.dst) return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[add.src1];
    _ = program.buffer_sizes[add.dst];
    return .{
        .input = mat.a,
        .weights = mat.b,
        .bias = add.src1,
        .output = add.dst,
        .temp = mat.dst,
        .k = @intCast(mat.geom.K),
        .n = @intCast(mat.geom.N),
    };
}

fn detectMatmul(program: backend_mod.DeviceProgram) ?MatmulShape {
    if (program.ops.len != 1) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const geom = mat.geom;
    if (geom.M == 0 or geom.N == 0 or geom.K == 0) return null;
    const m = u32Value(geom.M) orelse return null;
    const n = u32Value(geom.N) orelse return null;
    const k = u32Value(geom.K) orelse return null;
    const cells = std.math.mul(u32, m, n) catch return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[mat.dst];
    return .{
        .a = mat.a,
        .b = mat.b,
        .output = mat.dst,
        .m = m,
        .n = n,
        .k = k,
        .a_row_stride = u32Value(geom.a_row_stride) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .dst_row_stride = u32Value(geom.dst_row_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .dst_offset = u32Value(geom.dst_offset) orelse return null,
        .cells = cells,
    };
}

fn detectMatvecElementwise(program: backend_mod.DeviceProgram) ?MatvecElementwiseShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const ew = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!program_mod.matmulElementwiseSidecarCompatible(mat, ew)) return null;
    const geom = mat.geom;
    if (geom.M != 1 or geom.N == 0 or geom.K == 0) return null;
    const primary_is_src0 = ew.src0 == mat.dst and ew.src0_offset == geom.dst_offset;
    const secondary_buf = if (primary_is_src0) ew.src1 else ew.src0;
    const secondary_offset = if (primary_is_src0) ew.src1_offset else ew.src0_offset;
    if (secondary_buf == mat.dst) return null;
    if (ew.dst == mat.a or ew.dst == mat.b or ew.dst == secondary_buf) return null;
    const op_code = elementwiseOpCode(ew.op) orelse return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[secondary_buf];
    _ = program.buffer_sizes[ew.dst];
    return .{
        .input = mat.a,
        .weights = mat.b,
        .secondary = secondary_buf,
        .output = ew.dst,
        .op_code = op_code,
        .n = u32Value(geom.N) orelse return null,
        .k = u32Value(geom.K) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .secondary_offset = secondary_offset,
        .dst_offset = ew.dst_offset,
        .is_swapped = @intFromBool(!primary_is_src0),
    };
}

fn detectMatmulElementwise(program: backend_mod.DeviceProgram) ?MatmulElementwiseShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const ew = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!program_mod.matmulElementwiseSidecarCompatible(mat, ew)) return null;
    const geom = mat.geom;
    if (geom.M <= 1 or geom.N == 0 or geom.K == 0) return null;
    const m = u32Value(geom.M) orelse return null;
    const n = u32Value(geom.N) orelse return null;
    const k = u32Value(geom.K) orelse return null;
    const cells = std.math.mul(u32, m, n) catch return null;
    const primary_is_src0 = ew.src0 == mat.dst and ew.src0_offset == geom.dst_offset;
    const secondary_buf = if (primary_is_src0) ew.src1 else ew.src0;
    const secondary_offset = if (primary_is_src0) ew.src1_offset else ew.src0_offset;
    if (secondary_buf == mat.dst) return null;
    if (ew.dst == mat.a or ew.dst == mat.b or ew.dst == secondary_buf) return null;
    const op_code = elementwiseOpCode(ew.op) orelse return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[secondary_buf];
    _ = program.buffer_sizes[ew.dst];
    return .{
        .a = mat.a,
        .b = mat.b,
        .secondary = secondary_buf,
        .output = ew.dst,
        .op_code = op_code,
        .m = m,
        .n = n,
        .k = k,
        .a_row_stride = u32Value(geom.a_row_stride) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .secondary_offset = secondary_offset,
        .dst_offset = ew.dst_offset,
        .is_swapped = @intFromBool(!primary_is_src0),
        .cells = cells,
    };
}

fn detectMatvecFusedElementwise(program: backend_mod.DeviceProgram) ?MatvecFusedElementwiseShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const fe = switch (program.ops[1]) {
        .fused_elementwise => |f| f,
        else => return null,
    };
    const geom = mat.geom;
    if (geom.M != 1 or geom.N == 0 or geom.K == 0) return null;
    const dst_offset = u32Value(geom.dst_offset) orelse return null;
    if (geom.dst_row_stride != geom.N) return null;
    if (fe.n != geom.N) return null;
    if (fe.src != mat.dst or fe.src_offset != dst_offset) return null;
    if (fe.steps.len > max_fused_elementwise_steps) return null;
    if (fe.dst == mat.a or fe.dst == mat.b) return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[mat.dst];
    _ = program.buffer_sizes[fe.dst];

    var shape = MatvecFusedElementwiseShape{
        .input = mat.a,
        .weights = mat.b,
        .output = fe.dst,
        .secondary_bufs = .{mat.a} ** max_matvec_fused_elementwise_secondaries,
        .n = u32Value(geom.N) orelse return null,
        .k = u32Value(geom.K) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .dst_offset = fe.dst_offset,
        .n_steps = @intCast(fe.steps.len),
        .op = .{0} ** max_fused_elementwise_steps,
        .is_swapped = .{0} ** max_fused_elementwise_steps,
        .secondary_slot = .{0} ** max_fused_elementwise_steps,
        .secondary_offset = .{0} ** max_fused_elementwise_steps,
        .secondary_is_primary = .{0} ** max_fused_elementwise_steps,
    };
    var secondary_count: usize = 0;

    for (fe.steps, 0..) |step, i| {
        shape.op[i] = elementwiseOpCode(step.op) orelse return null;
        shape.is_swapped[i] = @intFromBool(step.is_swapped);
        shape.secondary_offset[i] = step.secondary_offset;
        if (step.op.isBinary()) {
            if (step.secondary_buf == mat.dst) {
                if (step.secondary_offset != dst_offset) return null;
                shape.secondary_is_primary[i] = 1;
                continue;
            }
            if (step.secondary_buf == fe.dst) return null;
            _ = program.buffer_sizes[step.secondary_buf];
            const slot = for (shape.secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                if (buf_idx == step.secondary_buf) break slot_idx;
            } else blk: {
                if (secondary_count >= max_matvec_fused_elementwise_secondaries) return null;
                const next = secondary_count;
                shape.secondary_bufs[next] = step.secondary_buf;
                secondary_count += 1;
                break :blk next;
            };
            shape.secondary_slot[i] = @intCast(slot);
        }
    }

    return shape;
}

fn detectMatmulFusedElementwise(program: backend_mod.DeviceProgram) ?MatmulFusedElementwiseShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const fe = switch (program.ops[1]) {
        .fused_elementwise => |f| f,
        else => return null,
    };
    const geom = mat.geom;
    if (geom.M <= 1 or geom.N == 0 or geom.K == 0) return null;
    if (!program_mod.matmulFusedElementwiseSidecarCompatible(mat, fe)) return null;
    if (fe.steps.len > max_fused_elementwise_steps) return null;
    if (fe.dst == mat.a or fe.dst == mat.b) return null;

    const m = u32Value(geom.M) orelse return null;
    const n = u32Value(geom.N) orelse return null;
    const k = u32Value(geom.K) orelse return null;
    const cells = std.math.mul(u32, m, n) catch return null;
    const primary_dst_offset = u32Value(geom.dst_offset) orelse return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[mat.dst];
    _ = program.buffer_sizes[fe.dst];

    var shape = MatmulFusedElementwiseShape{
        .a = mat.a,
        .b = mat.b,
        .output = fe.dst,
        .secondary_bufs = .{mat.a} ** max_matmul_fused_elementwise_secondaries,
        .m = m,
        .n = n,
        .k = k,
        .a_row_stride = u32Value(geom.a_row_stride) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .dst_row_stride = u32Value(geom.dst_row_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .primary_dst_offset = primary_dst_offset,
        .ew_dst_offset = fe.dst_offset,
        .n_steps = @intCast(fe.steps.len),
        .cells = cells,
        .op = .{0} ** max_fused_elementwise_steps,
        .is_swapped = .{0} ** max_fused_elementwise_steps,
        .secondary_slot = .{0} ** max_fused_elementwise_steps,
        .secondary_offset = .{0} ** max_fused_elementwise_steps,
        .secondary_is_primary = .{0} ** max_fused_elementwise_steps,
    };
    var secondary_count: usize = 0;

    for (fe.steps, 0..) |step, i| {
        shape.op[i] = elementwiseOpCode(step.op) orelse return null;
        shape.is_swapped[i] = @intFromBool(step.is_swapped);
        shape.secondary_offset[i] = step.secondary_offset;
        if (step.op.isBinary()) {
            if (step.secondary_buf == mat.dst) {
                if (step.secondary_offset != primary_dst_offset) return null;
                shape.secondary_is_primary[i] = 1;
                continue;
            }
            if (step.secondary_buf == fe.dst) return null;
            _ = program.buffer_sizes[step.secondary_buf];
            const slot = for (shape.secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                if (buf_idx == step.secondary_buf) break slot_idx;
            } else blk: {
                if (secondary_count >= max_matmul_fused_elementwise_secondaries) return null;
                const next = secondary_count;
                shape.secondary_bufs[next] = step.secondary_buf;
                secondary_count += 1;
                break :blk next;
            };
            shape.secondary_slot[i] = @intCast(slot);
        }
    }

    return shape;
}

fn detectMatvecSlice(program: backend_mod.DeviceProgram) ?MatvecSliceShape {
    if (program.ops.len != 2) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const slice = switch (program.ops[1]) {
        .slice_assign => |s| s,
        else => return null,
    };
    if (!program_mod.denseMatvecSliceSidecarCompatible(mat, slice)) return null;
    const geom = mat.geom;
    if (geom.M != 1 or geom.N == 0 or geom.K == 0) return null;
    if (slice.dst == mat.a or slice.dst == mat.b or slice.dst == mat.dst) return null;
    const cells = std.math.mul(u32, slice.rows, slice.cols) catch return null;
    if (cells == 0) return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[mat.dst];
    _ = program.buffer_sizes[slice.dst];
    return .{
        .input = mat.a,
        .weights = mat.b,
        .output = slice.dst,
        .n = u32Value(geom.N) orelse return null,
        .k = u32Value(geom.K) orelse return null,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .slice_src_col_start = program_mod.denseMatvecSliceSrcColStart(mat, slice).?,
        .slice_dst_offset = slice.dst_offset,
        .slice_dst_row_stride = slice.dst_row_stride,
        .slice_dst_col_stride = slice.dst_col_stride,
        .slice_rows = slice.rows,
        .cells = cells,
    };
}

fn detectMatvecRopeSlice(program: backend_mod.DeviceProgram) ?MatvecRopeSliceShape {
    if (program.ops.len != 3) return null;
    const mat = switch (program.ops[0]) {
        .matmul => |m| m,
        else => return null,
    };
    const rope = switch (program.ops[1]) {
        .rope => |r| r,
        else => return null,
    };
    const slice = switch (program.ops[2]) {
        .slice_assign => |s| s,
        else => return null,
    };
    if (!program_mod.denseMatvecRopeStoreSidecarCompatible(mat, rope, slice)) return null;
    const geom = mat.geom;
    if (geom.M != 1 or geom.N == 0 or geom.K == 0 or rope.half_d == 0) return null;
    if (slice.dst == mat.a or slice.dst == mat.b or slice.dst == mat.dst or slice.dst == rope.dst or slice.dst == rope.cos_sin) return null;
    _ = program.buffer_sizes[mat.a];
    _ = program.buffer_sizes[mat.b];
    _ = program.buffer_sizes[mat.dst];
    _ = program.buffer_sizes[rope.dst];
    _ = program.buffer_sizes[rope.cos_sin];
    _ = program.buffer_sizes[slice.dst];
    return .{
        .input = mat.a,
        .weights = mat.b,
        .cos_sin = rope.cos_sin,
        .output = slice.dst,
        .n = u32Value(geom.N) orelse return null,
        .k = u32Value(geom.K) orelse return null,
        .half_d = rope.half_d,
        .a_col_stride = u32Value(geom.a_col_stride) orelse return null,
        .b_row_stride = u32Value(geom.b_row_stride) orelse return null,
        .b_col_stride = u32Value(geom.b_col_stride) orelse return null,
        .a_offset = u32Value(geom.a_offset) orelse return null,
        .b_offset = u32Value(geom.b_offset) orelse return null,
        .rope_src_col_start = program_mod.denseMatvecRopeSrcColStart(mat, rope).?,
        .cs_off = rope.cs_off,
        .cs_cs = rope.cs_cs,
        .rope_dst_offset = slice.dst_offset,
        .rope_dst_row_stride = slice.dst_row_stride,
        .rope_dst_col_stride = slice.dst_col_stride,
    };
}

fn detectQMatmul(program: backend_mod.DeviceProgram) ?QMatmulShape {
    if (program.ops.len != 1) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    if (q.M == 0 or q.N == 0 or q.K == 0) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    const cells = std.math.mul(u32, q.M, q.N) catch return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[q.dst];
    return .{
        .input = q.input,
        .output = q.dst,
        .qweight_idx = q.weight_idx,
        .m = q.M,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K,
        .dst_offset = q.dst_offset,
        .dst_row_stride = if (q.dst_row_stride != 0) q.dst_row_stride else q.N,
        .cells = cells,
    };
}

fn detectQMatvecSlice(program: backend_mod.DeviceProgram) ?QMatvecSliceShape {
    if (program.ops.len != 2) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const slice = switch (program.ops[1]) {
        .slice_assign => |s| s,
        else => return null,
    };
    if (!program_mod.qmatvecSliceSidecarCompatible(q, slice)) return null;
    if (slice.dst == q.input or slice.dst == q.dst) return null;
    if (q.N == 0 or q.K == 0) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    const cells = std.math.mul(u32, slice.rows, slice.cols) catch return null;
    if (cells == 0) return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[q.dst];
    _ = program.buffer_sizes[slice.dst];
    return .{
        .input = q.input,
        .output = slice.dst,
        .qweight_idx = q.weight_idx,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .slice_src_col_start = program_mod.qmatmulSliceSrcColStart(q, slice).?,
        .slice_dst_offset = slice.dst_offset,
        .slice_dst_row_stride = slice.dst_row_stride,
        .slice_dst_col_stride = slice.dst_col_stride,
        .slice_rows = slice.rows,
        .cells = cells,
    };
}

fn detectQMatvecRopeSlice(program: backend_mod.DeviceProgram) ?QMatvecRopeSliceShape {
    if (program.ops.len != 3) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const rope = switch (program.ops[1]) {
        .rope => |r| r,
        else => return null,
    };
    const slice = switch (program.ops[2]) {
        .slice_assign => |s| s,
        else => return null,
    };
    if (!program_mod.qmatvecRopeStoreSidecarCompatible(q, rope, slice)) return null;
    if (slice.dst == q.input or slice.dst == q.dst or slice.dst == rope.dst or slice.dst == rope.cos_sin) return null;
    if (q.N == 0 or q.K == 0 or rope.half_d == 0) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[q.dst];
    _ = program.buffer_sizes[rope.dst];
    _ = program.buffer_sizes[rope.cos_sin];
    _ = program.buffer_sizes[slice.dst];
    return .{
        .input = q.input,
        .cos_sin = rope.cos_sin,
        .output = slice.dst,
        .qweight_idx = q.weight_idx,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .half_d = rope.half_d,
        .input_offset = q.input_offset,
        .rope_src_col_start = program_mod.qmatmulRopeSrcColStart(q, rope).?,
        .cs_off = rope.cs_off,
        .cs_cs = rope.cs_cs,
        .rope_dst_offset = slice.dst_offset,
        .rope_dst_row_stride = slice.dst_row_stride,
        .rope_dst_col_stride = slice.dst_col_stride,
    };
}

fn detectQLinear(program: backend_mod.DeviceProgram) ?QLinearShape {
    if (program.ops.len != 2) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const add = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (add.op != .add) return null;
    if (q.M != 1 or q.N == 0 or q.K == 0) return null;
    if (q.input_row_stride != 0 and q.input_row_stride != q.K) return null;
    if (q.dst_row_stride != 0 and q.dst_row_stride != q.N) return null;
    if (add.src0 != q.dst or add.src0_offset != q.dst_offset or add.n != q.N) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[add.src1];
    _ = program.buffer_sizes[add.dst];
    return .{
        .input = q.input,
        .bias = add.src1,
        .output = add.dst,
        .qweight_idx = q.weight_idx,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .bias_offset = add.src1_offset,
        .dst_offset = add.dst_offset,
    };
}

fn detectQMatmulElementwise(program: backend_mod.DeviceProgram) ?QMatmulElementwiseShape {
    if (program.ops.len != 2) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const ew = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!program_mod.qmatmulElementwiseSidecarCompatible(q, ew)) return null;
    if (q.M <= 1 or q.N == 0 or q.K == 0) return null;
    const q_is_src0 = ew.src0 == q.dst and ew.src0_offset == q.dst_offset;
    const secondary_buf = if (q_is_src0) ew.src1 else ew.src0;
    const secondary_offset = if (q_is_src0) ew.src1_offset else ew.src0_offset;
    if (secondary_buf == q.dst or ew.dst == q.input or ew.dst == secondary_buf) return null;
    const op_code = elementwiseOpCode(ew.op) orelse return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    const cells = std.math.mul(u32, q.M, q.N) catch return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[secondary_buf];
    _ = program.buffer_sizes[ew.dst];
    return .{
        .input = q.input,
        .secondary = secondary_buf,
        .output = ew.dst,
        .qweight_idx = q.weight_idx,
        .op_code = op_code,
        .m = q.M,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .input_row_stride = if (q.input_row_stride != 0) q.input_row_stride else q.K,
        .secondary_offset = secondary_offset,
        .dst_offset = ew.dst_offset,
        .is_swapped = @intFromBool(!q_is_src0),
        .cells = cells,
    };
}

fn detectQMatvecElementwise(program: backend_mod.DeviceProgram) ?QMatvecElementwiseShape {
    if (program.ops.len != 2) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const ew = switch (program.ops[1]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (!program_mod.qmatvecElementwiseSidecarCompatible(q, ew)) return null;
    const q_is_src0 = ew.src0 == q.dst and ew.src0_offset == q.dst_offset;
    const secondary_buf = if (q_is_src0) ew.src1 else ew.src0;
    const secondary_offset = if (q_is_src0) ew.src1_offset else ew.src0_offset;
    if (secondary_buf == q.dst or ew.dst == q.input or ew.dst == secondary_buf) return null;
    const op_code = elementwiseOpCode(ew.op) orelse return null;
    if (q.N == 0 or q.K == 0) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[secondary_buf];
    _ = program.buffer_sizes[ew.dst];
    return .{
        .input = q.input,
        .secondary = secondary_buf,
        .output = ew.dst,
        .qweight_idx = q.weight_idx,
        .op_code = op_code,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .secondary_offset = secondary_offset,
        .dst_offset = ew.dst_offset,
        .is_swapped = @intFromBool(!q_is_src0),
    };
}

fn detectQMatvecFusedElementwise(program: backend_mod.DeviceProgram) ?QMatvecFusedElementwiseShape {
    if (program.ops.len != 2) return null;
    const q = switch (program.ops[0]) {
        .qmatmul => |m| m,
        else => return null,
    };
    const fe = switch (program.ops[1]) {
        .fused_elementwise => |f| f,
        else => return null,
    };
    if (!program_mod.qmatvecFusedElementwiseSidecarCompatible(q, fe)) return null;
    if (q.N == 0 or q.K == 0) return null;
    if (fe.steps.len > max_fused_elementwise_steps) return null;
    if (fe.dst == q.input) return null;
    const qidx: usize = q.weight_idx;
    if (qidx >= program.qweights.len) return null;
    const qw = program.qweights[qidx];
    if (qw.rows != q.K or qw.cols != q.N) return null;
    const block_size = u32Value(qw.block_size) orelse return null;
    if (block_size == 0) return null;
    const n_data = std.math.mul(usize, qw.rows, qw.cols) catch return null;
    const n_blocks = if (n_data == 0) 0 else ((n_data - 1) / qw.block_size) + 1;
    if (qw.data.len < n_data or qw.scales.len < n_blocks) return null;
    _ = program.buffer_sizes[q.input];
    _ = program.buffer_sizes[q.dst];
    _ = program.buffer_sizes[fe.dst];

    var shape = QMatvecFusedElementwiseShape{
        .input = q.input,
        .output = fe.dst,
        .qweight_idx = q.weight_idx,
        .secondary_bufs = .{q.input} ** max_qmatvec_fused_elementwise_secondaries,
        .n = q.N,
        .k = q.K,
        .block_size = block_size,
        .input_offset = q.input_offset,
        .dst_offset = fe.dst_offset,
        .n_steps = @intCast(fe.steps.len),
        .op = .{0} ** max_fused_elementwise_steps,
        .is_swapped = .{0} ** max_fused_elementwise_steps,
        .secondary_slot = .{0} ** max_fused_elementwise_steps,
        .secondary_offset = .{0} ** max_fused_elementwise_steps,
        .secondary_is_primary = .{0} ** max_fused_elementwise_steps,
    };
    var secondary_count: usize = 0;

    for (fe.steps, 0..) |step, i| {
        shape.op[i] = elementwiseOpCode(step.op) orelse return null;
        shape.is_swapped[i] = @intFromBool(step.is_swapped);
        shape.secondary_offset[i] = step.secondary_offset;
        if (step.op.isBinary()) {
            if (step.secondary_buf == q.dst) {
                if (step.secondary_offset != q.dst_offset) return null;
                shape.secondary_is_primary[i] = 1;
                continue;
            }
            if (step.secondary_buf == fe.dst) return null;
            _ = program.buffer_sizes[step.secondary_buf];
            const slot = for (shape.secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                if (buf_idx == step.secondary_buf) break slot_idx;
            } else blk: {
                if (secondary_count >= max_qmatvec_fused_elementwise_secondaries) return null;
                const next = secondary_count;
                shape.secondary_bufs[next] = step.secondary_buf;
                secondary_count += 1;
                break :blk next;
            };
            shape.secondary_slot[i] = @intCast(slot);
        }
    }

    return shape;
}

fn elementwiseOpCode(op: backend_mod.Op) ?u32 {
    return switch (op) {
        .add => 1,
        .mul => 2,
        .neg => 3,
        .abs => 4,
        .sgn => 5,
        .step => 6,
        .relu => 7,
        .sqrt => 8,
        .recip => 9,
        .exp => 10,
        .log => 11,
        .gelu => 12,
        .sqr => 13,
        else => null,
    };
}

fn detectElementwise(program: backend_mod.DeviceProgram) ?ElementwiseShape {
    if (program.ops.len != 1) return null;
    const elem = switch (program.ops[0]) {
        .elementwise => |e| e,
        else => return null,
    };
    if (elem.n == 0) return null;
    _ = program.buffer_sizes[elem.src0];
    _ = program.buffer_sizes[elem.src1];
    _ = program.buffer_sizes[elem.dst];
    return .{
        .op_code = elementwiseOpCode(elem.op) orelse return null,
        .src0 = elem.src0,
        .src1 = elem.src1,
        .output = elem.dst,
        .n = elem.n,
        .dst_offset = elem.dst_offset,
        .src0_offset = elem.src0_offset,
        .src1_offset = elem.src1_offset,
    };
}

fn detectRepeatFusedElementwise(program: backend_mod.DeviceProgram) ?FusedElementwiseShape {
    if (program.ops.len != 2) return null;
    const rp = switch (program.ops[0]) {
        .repeat => |r| r,
        else => return null,
    };
    const fe = switch (program.ops[1]) {
        .fused_elementwise => |f| f,
        else => return null,
    };
    if (!program_mod.repeatFusedElementwiseCompatible(rp, fe)) return null;
    if (fe.n == 0 or fe.steps.len > max_fused_elementwise_steps) return null;
    if (rp.src == fe.dst) return null;
    _ = program.buffer_sizes[rp.src];
    _ = program.buffer_sizes[rp.dst];
    _ = program.buffer_sizes[fe.src];
    _ = program.buffer_sizes[fe.dst];

    var shape = FusedElementwiseShape{
        .src = fe.src,
        .output = fe.dst,
        .secondary_bufs = .{fe.src} ** max_fused_elementwise_secondaries,
        .n = fe.n,
        .n_steps = @intCast(fe.steps.len),
        .dst_offset = fe.dst_offset,
        .src_offset = fe.src_offset,
        .op = .{0} ** max_fused_elementwise_steps,
        .is_swapped = .{0} ** max_fused_elementwise_steps,
        .secondary_slot = .{0} ** max_fused_elementwise_steps,
        .secondary_offset = .{0} ** max_fused_elementwise_steps,
        .secondary_is_repeat = .{0} ** max_fused_elementwise_steps,
        .secondary_repeat_dst_offset = .{0} ** max_fused_elementwise_steps,
        .repeat_src_offset = rp.src_offset,
        .repeat_src_ne = rp.src_ne,
        .repeat_src_strides = rp.src_strides,
        .repeat_dst_strides = rp.dst_strides,
    };
    var secondary_count: usize = 0;

    for (fe.steps, 0..) |step, i| {
        shape.op[i] = elementwiseOpCode(step.op) orelse return null;
        shape.is_swapped[i] = @intFromBool(step.is_swapped);
        if (step.op.isBinary()) {
            var secondary_buf = step.secondary_buf;
            if (step.secondary_buf == rp.dst) {
                if (step.secondary_offset < rp.dst_offset) return null;
                const rel = step.secondary_offset - rp.dst_offset;
                if (@as(u64, rel) + @as(u64, fe.n) > @as(u64, rp.n)) return null;
                shape.secondary_is_repeat[i] = 1;
                shape.secondary_repeat_dst_offset[i] = rel;
                secondary_buf = rp.src;
            } else {
                shape.secondary_offset[i] = step.secondary_offset;
            }
            _ = program.buffer_sizes[secondary_buf];
            const slot = for (shape.secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                if (buf_idx == secondary_buf) break slot_idx;
            } else blk: {
                if (secondary_count >= max_fused_elementwise_secondaries) return null;
                const next = secondary_count;
                shape.secondary_bufs[next] = secondary_buf;
                secondary_count += 1;
                break :blk next;
            };
            shape.secondary_slot[i] = @intCast(slot);
        }
    }

    return shape;
}

fn detectRepeat(program: backend_mod.DeviceProgram) ?RepeatShape {
    if (program.ops.len != 1) return null;
    const rp = switch (program.ops[0]) {
        .repeat => |r| r,
        else => return null,
    };
    if (rp.n == 0) return null;
    _ = program.buffer_sizes[rp.src];
    _ = program.buffer_sizes[rp.dst];
    return .{
        .src = rp.src,
        .output = rp.dst,
        .n = rp.n,
        .src_offset = rp.src_offset,
        .dst_offset = rp.dst_offset,
        .src_ne = rp.src_ne,
        .src_strides = rp.src_strides,
        .dst_strides = rp.dst_strides,
    };
}

fn detectFusedElementwise(program: backend_mod.DeviceProgram) ?FusedElementwiseShape {
    if (program.ops.len != 1) return null;
    const fe = switch (program.ops[0]) {
        .fused_elementwise => |f| f,
        else => return null,
    };
    if (fe.n == 0 or fe.steps.len > max_fused_elementwise_steps) return null;
    _ = program.buffer_sizes[fe.src];
    _ = program.buffer_sizes[fe.dst];

    var shape = FusedElementwiseShape{
        .src = fe.src,
        .output = fe.dst,
        .secondary_bufs = .{fe.src} ** max_fused_elementwise_secondaries,
        .n = fe.n,
        .n_steps = @intCast(fe.steps.len),
        .dst_offset = fe.dst_offset,
        .src_offset = fe.src_offset,
        .op = .{0} ** max_fused_elementwise_steps,
        .is_swapped = .{0} ** max_fused_elementwise_steps,
        .secondary_slot = .{0} ** max_fused_elementwise_steps,
        .secondary_offset = .{0} ** max_fused_elementwise_steps,
        .secondary_is_repeat = .{0} ** max_fused_elementwise_steps,
        .secondary_repeat_dst_offset = .{0} ** max_fused_elementwise_steps,
        .repeat_src_offset = 0,
        .repeat_src_ne = .{ 1, 1, 1, 1 },
        .repeat_src_strides = .{0} ** 4,
        .repeat_dst_strides = .{0} ** 4,
    };
    var secondary_count: usize = 0;

    for (fe.steps, 0..) |step, i| {
        shape.op[i] = elementwiseOpCode(step.op) orelse return null;
        shape.is_swapped[i] = @intFromBool(step.is_swapped);
        shape.secondary_offset[i] = step.secondary_offset;
        if (step.op.isBinary()) {
            _ = program.buffer_sizes[step.secondary_buf];
            const slot = for (shape.secondary_bufs[0..secondary_count], 0..) |buf_idx, slot_idx| {
                if (buf_idx == step.secondary_buf) break slot_idx;
            } else blk: {
                if (secondary_count >= max_fused_elementwise_secondaries) return null;
                const next = secondary_count;
                shape.secondary_bufs[next] = step.secondary_buf;
                secondary_count += 1;
                break :blk next;
            };
            shape.secondary_slot[i] = @intCast(slot);
        }
    }

    return shape;
}

fn detectLayerNorm(program: backend_mod.DeviceProgram) ?LayerNormShape {
    if (program.ops.len != 1) return null;
    const ln = switch (program.ops[0]) {
        .layernorm => |l| l,
        else => return null,
    };
    if (ln.rows == 0 or ln.cols == 0) return null;
    _ = program.buffer_sizes[ln.src];
    _ = program.buffer_sizes[ln.dst];
    return .{
        .src = ln.src,
        .output = ln.dst,
        .rows = ln.rows,
        .cols = ln.cols,
        .eps = ln.eps,
        .src_offset = ln.src_offset,
        .dst_offset = ln.dst_offset,
    };
}

fn detectRmsNorm(program: backend_mod.DeviceProgram) ?RmsNormShape {
    if (program.ops.len != 1) return null;
    const rms = switch (program.ops[0]) {
        .rmsnorm => |r| r,
        else => return null,
    };
    if (rms.rows == 0 or rms.cols == 0) return null;
    _ = program.buffer_sizes[rms.src];
    _ = program.buffer_sizes[rms.dst];
    return .{
        .src = rms.src,
        .output = rms.dst,
        .rows = rms.rows,
        .cols = rms.cols,
        .eps = rms.eps,
        .src_offset = rms.src_offset,
        .dst_offset = rms.dst_offset,
    };
}

fn detectSoftmax(program: backend_mod.DeviceProgram) ?SoftmaxShape {
    if (program.ops.len != 1) return null;
    const soft = switch (program.ops[0]) {
        .softmax => |s| s,
        else => return null,
    };
    if (soft.rows == 0 or soft.cols == 0) return null;
    _ = std.math.mul(u32, soft.rows, soft.cols) catch return null;
    _ = program.buffer_sizes[soft.src];
    _ = program.buffer_sizes[soft.dst];
    return .{
        .src = soft.src,
        .output = soft.dst,
        .rows = soft.rows,
        .cols = soft.cols,
        .src_offset = soft.src_offset,
        .dst_offset = soft.dst_offset,
    };
}

fn detectLogSoftmax(program: backend_mod.DeviceProgram) ?SoftmaxShape {
    if (program.ops.len != 1) return null;
    const soft = switch (program.ops[0]) {
        .logsoftmax => |s| s,
        else => return null,
    };
    if (soft.rows == 0 or soft.cols == 0) return null;
    _ = std.math.mul(u32, soft.rows, soft.cols) catch return null;
    _ = program.buffer_sizes[soft.src];
    _ = program.buffer_sizes[soft.dst];
    return .{
        .src = soft.src,
        .output = soft.dst,
        .rows = soft.rows,
        .cols = soft.cols,
        .src_offset = soft.src_offset,
        .dst_offset = soft.dst_offset,
    };
}

fn reduceOpCode(op: backend_mod.Op) ?u32 {
    return switch (op) {
        .sum => 1,
        .max => 2,
        .min => 3,
        else => null,
    };
}

fn detectReduce(program: backend_mod.DeviceProgram) ?ReduceShape {
    if (program.ops.len != 1) return null;
    const reduce = switch (program.ops[0]) {
        .reduce => |r| r,
        else => return null,
    };
    if (reduce.n_out == 0 or reduce.reduce_size == 0) return null;
    _ = std.math.mul(u32, reduce.n_out, reduce.reduce_size) catch return null;
    _ = program.buffer_sizes[reduce.src];
    _ = program.buffer_sizes[reduce.dst];
    return .{
        .op_code = reduceOpCode(reduce.op) orelse return null,
        .src = reduce.src,
        .output = reduce.dst,
        .n_out = reduce.n_out,
        .reduce_size = reduce.reduce_size,
        .src_offset = reduce.src_offset,
        .dst_offset = reduce.dst_offset,
    };
}

fn detectRope(program: backend_mod.DeviceProgram) ?RopeShape {
    if (program.ops.len != 1) return null;
    const rope = switch (program.ops[0]) {
        .rope => |r| r,
        else => return null,
    };
    if (rope.half_d == 0 or rope.seq_len == 0) return null;
    const cells = std.math.mul(u32, rope.half_d, rope.seq_len) catch return null;
    _ = program.buffer_sizes[rope.src];
    _ = program.buffer_sizes[rope.cos_sin];
    _ = program.buffer_sizes[rope.dst];
    return .{
        .src = rope.src,
        .cos_sin = rope.cos_sin,
        .output = rope.dst,
        .half_d = rope.half_d,
        .seq_len = rope.seq_len,
        .src_off = rope.src_off,
        .cs_off = rope.cs_off,
        .dst_off = rope.dst_off,
        .src_rs = rope.src_rs,
        .src_cs = rope.src_cs,
        .cs_cs = rope.cs_cs,
        .cells = cells,
    };
}

fn detectSliceAssign(program: backend_mod.DeviceProgram) ?SliceAssignShape {
    if (program.ops.len != 1) return null;
    const slice = switch (program.ops[0]) {
        .slice_assign => |s| s,
        else => return null,
    };
    if (slice.rows == 0 or slice.cols == 0) return null;
    const cells = std.math.mul(u32, slice.rows, slice.cols) catch return null;
    _ = program.buffer_sizes[slice.src];
    _ = program.buffer_sizes[slice.dst];
    return .{
        .src = slice.src,
        .output = slice.dst,
        .rows = slice.rows,
        .cols = slice.cols,
        .dst_offset = slice.dst_offset,
        .dst_row_stride = slice.dst_row_stride,
        .dst_col_stride = slice.dst_col_stride,
        .src_offset = slice.src_offset,
        .src_row_stride = slice.src_row_stride,
        .src_col_stride = slice.src_col_stride,
        .cells = cells,
    };
}

fn detectAttention(program: backend_mod.DeviceProgram) ?AttentionShape {
    if (program.ops.len != 1) return null;
    const att = switch (program.ops[0]) {
        .attention => |a| a,
        else => return null,
    };
    if (att.d_head == 0 or att.seq_q == 0 or att.seq_kv == 0) return null;
    if (att.d_head > max_attention_d_head or att.seq_kv > max_attention_seq_kv) return null;
    if (att.q_rs == 0 or att.q_cs == 0 or att.k_rs == 0 or att.k_cs == 0) return null;
    if (att.v_rs == 0 or att.v_cs == 0 or att.dst_rs == 0 or att.dst_cs == 0) return null;
    if (att.has_mask and (att.mask_rs == 0 or att.mask_cs == 0)) return null;
    _ = program.buffer_sizes[att.q];
    _ = program.buffer_sizes[att.k];
    _ = program.buffer_sizes[att.v];
    _ = program.buffer_sizes[att.mask];
    _ = program.buffer_sizes[att.dst];
    return .{
        .q = att.q,
        .k = att.k,
        .v = att.v,
        .mask = att.mask,
        .output = att.dst,
        .d_head = att.d_head,
        .seq_q = att.seq_q,
        .seq_kv = att.seq_kv,
        .has_mask = @intFromBool(att.has_mask),
        .scale = att.scale,
        .q_off = att.q_off,
        .k_off = att.k_off,
        .v_off = att.v_off,
        .mask_off = att.mask_off,
        .dst_off = att.dst_off,
        .q_rs = att.q_rs,
        .q_cs = att.q_cs,
        .k_rs = att.k_rs,
        .k_cs = att.k_cs,
        .v_rs = att.v_rs,
        .v_cs = att.v_cs,
        .mask_rs = att.mask_rs,
        .mask_cs = att.mask_cs,
        .dst_rs = att.dst_rs,
        .dst_cs = att.dst_cs,
    };
}

fn createPipeline(gpu: *GpuContext, shape: ProgramShape) !PipelineResources {
    var shader_source: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
    shader_source.chain.sType = c.WGPUSType_ShaderSourceWGSL;
    shader_source.code = stringView(shape.shaderSource());
    var shader_desc: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
    shader_desc.nextInChain = &shader_source.chain;
    const shader = c.wgpuDeviceCreateShaderModule(gpu.device, &shader_desc) orelse return error.WgpuUnavailable;
    defer c.wgpuShaderModuleRelease(shader);

    var pipeline_desc: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = stringView("main");
    const pipeline = c.wgpuDeviceCreateComputePipeline(gpu.device, &pipeline_desc) orelse return error.WgpuUnavailable;
    errdefer c.wgpuComputePipelineRelease(pipeline);
    const bind_group_layout = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0) orelse return error.WgpuUnavailable;
    errdefer c.wgpuBindGroupLayoutRelease(bind_group_layout);

    return .{
        .pipeline = pipeline,
        .bind_group_layout = bind_group_layout,
    };
}

fn compileDispatches(alloc: std.mem.Allocator, gpu: *GpuContext, program: backend_mod.DeviceProgram) ![]CompiledDispatch {
    const dispatch_count = countDispatches(program) orelse return error.UnsupportedDeviceOp;
    const dispatches = try alloc.alloc(CompiledDispatch, dispatch_count);
    var made: usize = 0;
    errdefer {
        for (dispatches[0..made]) |dispatch| {
            c.wgpuBindGroupLayoutRelease(dispatch.bind_group_layout);
            c.wgpuComputePipelineRelease(dispatch.pipeline);
        }
        alloc.free(dispatches);
    }

    var op_index: usize = 0;
    while (op_index < program.ops.len) {
        const detected = detectDispatchAt(program, op_index) orelse return error.UnsupportedDeviceOp;
        const pipeline = try createPipeline(gpu, detected.shape);
        dispatches[made] = .{
            .shape = detected.shape,
            .op_start = op_index,
            .op_count = detected.op_count,
            .pipeline = pipeline.pipeline,
            .bind_group_layout = pipeline.bind_group_layout,
        };
        made += 1;
        op_index += detected.op_count;
    }
    std.debug.assert(made == dispatch_count);
    return dispatches;
}

const ClonedInitialUploads = struct {
    uploads: []backend_mod.ProgramIO,
    bytes: []u8,
};

fn cloneInitialUploads(alloc: std.mem.Allocator, uploads: []const backend_mod.ProgramIO) !ClonedInitialUploads {
    if (uploads.len == 0) return .{ .uploads = &.{}, .bytes = &.{} };
    var total_bytes: usize = 0;
    for (uploads) |io| {
        if (!io.isHost()) return error.UnsupportedResourceBinding;
        total_bytes = std.math.add(usize, total_bytes, io.size) catch return error.InvalidProgramIO;
    }

    const cloned_uploads = try alloc.alloc(backend_mod.ProgramIO, uploads.len);
    errdefer alloc.free(cloned_uploads);
    const bytes = try alloc.alloc(u8, total_bytes);
    errdefer alloc.free(bytes);

    var offset: usize = 0;
    for (uploads, cloned_uploads) |io, *cloned| {
        const len: usize = io.size;
        @memcpy(bytes[offset..][0..len], io.host_ptr[0..len]);
        cloned.* = backend_mod.ProgramIO.host(io.buf_idx, io.offset, bytes[offset..].ptr, io.size);
        offset += len;
    }
    return .{ .uploads = cloned_uploads, .bytes = bytes };
}

fn deinitClonedInitialUploads(alloc: std.mem.Allocator, cloned: ClonedInitialUploads) void {
    if (cloned.uploads.len > 0) alloc.free(cloned.uploads);
    if (cloned.bytes.len > 0) alloc.free(cloned.bytes);
}

fn compileProgramInner(alloc: std.mem.Allocator, gpu: GpuContext, program: backend_mod.DeviceProgram) !*CompiledProgram {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    var owned_gpu = gpu;
    errdefer owned_gpu.deinit();

    const dispatches = try compileDispatches(alloc, &owned_gpu, program);
    errdefer releaseDispatches(alloc, dispatches);

    const qweights = try prepareDeviceQWeights(&owned_gpu, alloc, program.qweights);
    errdefer releaseDeviceQWeights(alloc, qweights);
    const initial_uploads = try cloneInitialUploads(alloc, program.initial_uploads);
    errdefer deinitClonedInitialUploads(alloc, initial_uploads);

    const compiled = try alloc.create(CompiledProgram);
    errdefer alloc.destroy(compiled);
    var program_stencil = try program_mod.ProgramStencil.initProgramWithKernelizer(alloc, program, program_mod.Kernelizer.default());
    errdefer program_stencil.deinit(alloc);
    const buffer_sizes = try alloc.dupe(usize, program.buffer_sizes);
    errdefer alloc.free(buffer_sizes);

    compiled.* = .{
        .gpu = owned_gpu,
        .dispatches = dispatches,
        .program_stencil = program_stencil,
        .qweights = qweights,
        .default_runtime = undefined,
        .buffer_sizes = buffer_sizes,
        .initial_uploads = initial_uploads.uploads,
        .initial_upload_bytes = initial_uploads.bytes,
        .registered_resources = .empty,
        .alloc = alloc,
    };
    compiled.default_runtime = try RuntimeBindings.init(alloc, compiled);
    errdefer compiled.default_runtime.deinit();
    return compiled;
}

pub fn createDeviceBuffer(handle: backend_mod.Backend.CompiledHandle, byte_len: usize, access: backend_mod.ProgramIO.ExternalResource.Access) !backend_mod.ProgramIO.ExternalResource {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    if (byte_len == 0 or byte_len > std.math.maxInt(u32)) return error.InvalidProgramIO;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const buffer = try createBuffer(
        compiled.gpu.device,
        @intCast(byte_len),
        c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc,
    );
    errdefer c.wgpuBufferRelease(buffer);
    var resource = try compiled.registerExternalBuffer(buffer, @intCast(byte_len));
    compiled.registered_resources.items[compiled.registered_resources.items.len - 1].release_on_drop = true;
    resource.access = access;
    return resource;
}

pub fn releaseDeviceBuffer(handle: backend_mod.Backend.CompiledHandle, resource: backend_mod.ProgramIO.ExternalResource) void {
    if (!options.use_wgpu) return;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const removed = compiled.unregisterExternalBuffer(resource) orelse return;
    if (removed.release_on_drop) c.wgpuBufferRelease(removed.buffer);
}

pub fn deviceHandle(handle: backend_mod.Backend.CompiledHandle) !usize {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return deviceHandleValue(compiled.gpu.device) orelse error.WgpuUnavailable;
}

pub fn importDeviceBuffer(
    handle: backend_mod.Backend.CompiledHandle,
    device_handle: usize,
    buffer_handle: usize,
    byte_offset: usize,
    view_byte_len: usize,
    access: backend_mod.ProgramIO.ExternalResource.Access,
) !backend_mod.ProgramIO.ExternalResource {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    if (buffer_handle == 0 or view_byte_len == 0) return error.InvalidProgramIO;
    if (view_byte_len > std.math.maxInt(u32) or byte_offset > std.math.maxInt(u32)) return error.InvalidProgramIO;
    if ((@as(u64, @intCast(byte_offset)) % storage_buffer_binding_alignment) != 0) return error.UnsupportedResourceBinding;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const expected_device = deviceHandleValue(compiled.gpu.device) orelse return error.WgpuUnavailable;
    if (device_handle != expected_device) return error.UnsupportedResourceBinding;
    const raw_buffer: *c.struct_WGPUBufferImpl = @ptrFromInt(buffer_handle);
    const buffer: c.WGPUBuffer = raw_buffer;
    const actual_byte_len = c.wgpuBufferGetSize(buffer);
    const end = std.math.add(u64, @intCast(byte_offset), @intCast(view_byte_len)) catch return error.InvalidProgramIO;
    if (end > actual_byte_len or actual_byte_len > std.math.maxInt(u32)) return error.InvalidProgramIO;

    var required_usage: c.WGPUBufferUsage = c.WGPUBufferUsage_Storage;
    if (access.read) required_usage |= c.WGPUBufferUsage_CopyDst;
    if (access.write) required_usage |= c.WGPUBufferUsage_CopySrc;
    const usage = c.wgpuBufferGetUsage(buffer);
    if ((usage & required_usage) != required_usage) return error.UnsupportedResourceBinding;

    c.wgpuBufferAddRef(buffer);
    errdefer c.wgpuBufferRelease(buffer);
    var resource = try compiled.registerExternalBuffer(buffer, actual_byte_len);
    compiled.registered_resources.items[compiled.registered_resources.items.len - 1].release_on_drop = true;
    resource.byte_offset = @intCast(byte_offset);
    resource.access = access;
    return resource;
}

pub fn writeDeviceBuffer(handle: backend_mod.Backend.CompiledHandle, resource: backend_mod.ProgramIO.ExternalResource, byte_offset: usize, src: []const u8) !void {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
    const begin = std.math.add(u64, resource.byte_offset, std.math.cast(u64, byte_offset) orelse return error.InvalidProgramIO) catch return error.InvalidProgramIO;
    if (begin > registered.byte_len or src.len > registered.byte_len - begin) return error.InvalidProgramIO;
    if (src.len == 0) return;
    c.wgpuQueueWriteBuffer(compiled.gpu.queue, registered.buffer, begin, src.ptr, src.len);
}

pub fn readDeviceBuffer(handle: backend_mod.Backend.CompiledHandle, resource: backend_mod.ProgramIO.ExternalResource, byte_offset: usize, dst: []u8) !void {
    if (!options.use_wgpu) return error.WgpuUnavailable;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const registered = compiled.resolveExternalBuffer(resource) orelse return error.UnsupportedResourceBinding;
    const begin = std.math.add(u64, resource.byte_offset, std.math.cast(u64, byte_offset) orelse return error.InvalidProgramIO) catch return error.InvalidProgramIO;
    if (begin > registered.byte_len or dst.len > registered.byte_len - begin) return error.InvalidProgramIO;
    if (dst.len == 0) return;
    try readBufferBytes(&compiled.gpu, registered.buffer, begin, dst);
}

fn wgpuBackend(ctx: *anyopaque) *WgpuBackend {
    return @ptrCast(@alignCast(ctx));
}

fn denseMatMulF32(_: *anyopaque, _: backend_mod.DenseMatMulSpecF32) bool {
    return false;
}

fn supportsProgram(_: *anyopaque, program: backend_mod.DeviceProgram) bool {
    return supportsDispatchPlan(program);
}

fn compileProgram(ctx: *anyopaque, program: backend_mod.DeviceProgram) ?backend_mod.Backend.CompiledHandle {
    const backend = wgpuBackend(ctx);
    const gpu = backend.acquireGpu() catch return null;
    const compiled = compileProgramInner(backend.alloc, gpu, program) catch {
        return null;
    };
    return @ptrCast(compiled);
}

fn inspectExecutionPlan(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) backend_mod.ExecutionPlanInspection {
    return inspectCompiledDispatchPlan(handle);
}

fn runtimeBindings(runtime: backend_mod.Backend.RuntimeHandle) *RuntimeBindings {
    return @ptrCast(@alignCast(runtime));
}

fn bindProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) ?backend_mod.Backend.RuntimeHandle {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const runtime = compiled.alloc.create(RuntimeBindings) catch return null;
    runtime.* = RuntimeBindings.init(compiled.alloc, compiled) catch {
        compiled.alloc.destroy(runtime);
        return null;
    };
    return @ptrCast(runtime);
}

fn configureBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, persistent_inputs: []const backend_mod.ProgramIO, step_inputs: []const backend_mod.ProgramIO, step_outputs: []const backend_mod.ProgramIO) bool {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    runtimeBindings(runtime).configure(compiled, persistent_inputs, step_inputs, step_outputs) catch return false;
    return true;
}

fn patchRuntimeBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(runtimeBindings(runtime), window);
}

fn uploadBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.upload(runtimeBindings(runtime), inputs);
}

fn uploadQWeights(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, qweights: []const backend_mod.QuantizedWeightUpload) bool {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    runtimeBindings(runtime).uploadQWeights(compiled, qweights) catch return false;
    return true;
}

fn executeBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(runtimeBindings(runtime), inputs, outputs, true);
}

fn executeConfiguredBindings(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, download_outputs: bool) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const bindings = runtimeBindings(runtime);
    compiled.execute(bindings, bindings.configured_inputs, bindings.configured_outputs, download_outputs);
}

fn freeBindings(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle) void {
    const binding = runtimeBindings(runtime);
    const alloc = binding.alloc;
    binding.deinit();
    alloc.destroy(binding);
}

fn patchRuntimeWindow(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, window: backend_mod.RuntimeWindow) backend_mod.RuntimePatchStatus {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    return compiled.patchRuntimeWindow(&compiled.default_runtime, window);
}

fn uploadProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.upload(&compiled.default_runtime, inputs);
}

fn executeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, inputs: []const backend_mod.ProgramIO, outputs: []const backend_mod.ProgramIO) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.execute(&compiled.default_runtime, inputs, outputs, true);
}

fn freeProgram(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.deinit();
}

fn resetRuntimeProfile(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    compiled.default_runtime.runtime_profile.reset();
}

fn addRuntimeProfileTo(_: *anyopaque, handle: backend_mod.Backend.CompiledHandle, dest: *profile_mod.RuntimeProfile) void {
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    dest.add(compiled.default_runtime.runtime_profile);
}

fn resetRuntimeBindingsProfile(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle) void {
    runtimeBindings(runtime).runtime_profile.reset();
}

fn addRuntimeBindingsProfileTo(_: *anyopaque, _: backend_mod.Backend.CompiledHandle, runtime: backend_mod.Backend.RuntimeHandle, dest: *profile_mod.RuntimeProfile) void {
    dest.add(runtimeBindings(runtime).runtime_profile);
}

const vtable = backend_mod.Backend.VTable{
    .dense_matmul_f32 = denseMatMulF32,
    .supports_program = supportsProgram,
    .compile_program = compileProgram,
    .inspect_execution_plan = inspectExecutionPlan,
    .bind_program = bindProgram,
    .configure_bindings = configureBindings,
    .patch_runtime_bindings = patchRuntimeBindings,
    .upload_bindings = uploadBindings,
    .upload_qweights = uploadQWeights,
    .execute_bindings = executeBindings,
    .execute_configured_bindings = executeConfiguredBindings,
    .free_bindings = freeBindings,
    .patch_runtime_window = patchRuntimeWindow,
    .upload_program = uploadProgram,
    .execute_program = executeProgram,
    .free_program = freeProgram,
    .reset_runtime_profile = resetRuntimeProfile,
    .add_runtime_profile_to = addRuntimeProfileTo,
    .reset_runtime_bindings_profile = resetRuntimeBindingsProfile,
    .add_runtime_bindings_profile_to = addRuntimeBindingsProfileTo,
};

test "wgpu backend executes zero-dispatch program" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();
    try std.testing.expect(be.capabilities.executes_programs);
    try std.testing.expect(be.capabilities.external_resources);

    const buffer_sizes = [_]usize{4};
    var initial = [_]f32{ 0, 0, 0, 0 };
    const uploads = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&initial), initial.len * @sizeOf(f32)),
    };
    const program = backend_mod.DeviceProgram{
        .ops = &.{},
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
    };
    try std.testing.expect(be.supportsProgram(program));

    const plan = inspectDispatchPlan(program);
    try std.testing.expect(plan.supported);
    try std.testing.expectEqual(@as(u64, 0), plan.total_op_count);
    try std.testing.expectEqual(@as(u64, 0), plan.covered_op_count);
    try std.testing.expectEqual(@as(u64, 0), plan.dispatch_count);
    try std.testing.expectEqual(@as(?u64, null), plan.first_unsupported_op);

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);

    const compiled_plan = be.inspectExecutionPlan(handle);
    try std.testing.expect(compiled_plan.supported);
    try std.testing.expectEqual(@as(u64, 0), compiled_plan.total_op_count);
    try std.testing.expectEqual(@as(u64, 0), compiled_plan.covered_op_count);
    try std.testing.expectEqual(@as(u64, 0), compiled_plan.dispatch_count);
    try std.testing.expectEqual(@as(?u64, null), compiled_plan.first_unsupported_op);

    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 8, 6, 4, 2 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&output), output.len * @sizeOf(f32))};
    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectEqualSlices(f32, input[0..], output[0..]);

    var host_rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &host_rt);
    try std.testing.expectEqual(@as(u32, 1), host_rt.call_count);
    try std.testing.expectEqual(@as(u32, 0), host_rt.program_command_shape.command_count);
    try std.testing.expectEqual(@as(u64, 0), host_rt.program_command_shape.command_stencil_hash);
    try std.testing.expectEqual(@as(u64, 0), host_rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), host_rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), host_rt.sync_count);

    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_input);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(output.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 9, 7, 5, 3 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(output.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(0, 0, output_resource, output.len * @sizeOf(f32))};
    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.resetRuntimeBindingsProfile(handle, runtime);
    be.executeConfiguredBindings(handle, runtime, false);

    var resource_output_values = [_]f32{ 0, 0, 0, 0 };
    try readBufferF32(&compiled.gpu, external_output, &resource_output_values);
    try std.testing.expectEqualSlices(f32, resource_input_values[0..], resource_output_values[0..]);

    var resource_rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &resource_rt);
    try std.testing.expectEqual(@as(u32, 1), resource_rt.call_count);
    try std.testing.expectEqual(@as(u32, 0), resource_rt.program_command_shape.command_count);
    try std.testing.expectEqual(@as(u64, 0), resource_rt.program_command_shape.command_stencil_hash);
    try std.testing.expectEqual(@as(u64, 0), resource_rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 0), resource_rt.backend_dispatch_count);
    try std.testing.expectEqual(@as(u64, 0), resource_rt.sync_count);
}

test "wgpu backend executes host-staged tiny linear program" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();
    try std.testing.expect(be.capabilities.executes_programs);
    try std.testing.expect(be.capabilities.external_resources);

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 3,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 3,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 3,
            },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 4, .src0 = 3, .src1 = 2, .n = 3 } },
    };
    const buffer_sizes = [_]usize{ 2, 6, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var bias = [_]f32{ 0.5, -1, 2 };
    var output = [_]f32{ 0, 0, 0 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&bias), bias.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);

    try std.testing.expectApproxEqAbs(@as(f32, 14.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 18), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 26), output[2], 0.001);

    weights = [_]f32{
        9, 9, 9,
        9, 9, 9,
    };
    input = [_]f32{ 4, 5 };
    output = [_]f32{ 0, 0, 0 };
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 24.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 32), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44), output[2], 0.001);

    input = [_]f32{ 6, 7 };
    output = [_]f32{ -5, -5, -5 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), output[2], 0.001);

    const fake_weights = [_]backend_mod.ProgramIO{
        .external(1, 0, .{
            .placement = .webgpu,
            .handle = 999_999,
            .byte_len = @intCast(weights.len * @sizeOf(f32)),
            .access = .read_only,
        }, weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(bias[0..].ptr), bias.len * @sizeOf(f32)),
    };
    try std.testing.expect(!be.configureBindings(handle, runtime, &fake_weights, &inputs, &outputs));

    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_bias = try createBuffer(compiled.gpu.device, try bytesForElements(bias.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_bias);
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(output.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    var resource_weights_values = [_]f32{
        1, 0, 0,
        0, 1, 0,
    };
    var resource_bias_values = [_]f32{ 0, 0, 10 };
    var resource_input_values = [_]f32{ 8, 9 };
    try writeBufferF32(&compiled.gpu, external_weights, resource_weights_values[0..]);
    try writeBufferF32(&compiled.gpu, external_bias, resource_bias_values[0..]);
    try writeBufferF32(&compiled.gpu, external_input, resource_input_values[0..]);

    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var bias_resource = try compiled.registerExternalBuffer(external_bias, try bytesForElements(resource_bias_values.len));
    bias_resource.access = .read_only;
    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(output.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, bias_resource, resource_bias_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{
        .external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32)),
    };
    const resource_outputs = [_]backend_mod.ProgramIO{
        .external(4, 0, output_resource, output.len * @sizeOf(f32)),
    };
    var write_only_weights = weights_resource;
    write_only_weights.access = .write_only;
    const bad_persistent_access = [_]backend_mod.ProgramIO{
        .external(1, 0, write_only_weights, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, bias_resource, resource_bias_values.len * @sizeOf(f32)),
    };
    try std.testing.expect(!be.configureBindings(handle, runtime, &bad_persistent_access, &resource_inputs, &resource_outputs));

    var write_only_input = input_resource;
    write_only_input.access = .write_only;
    const bad_input_access = [_]backend_mod.ProgramIO{
        .external(0, 0, write_only_input, resource_input_values.len * @sizeOf(f32)),
    };
    try std.testing.expect(!be.configureBindings(handle, runtime, &resource_persistent, &bad_input_access, &resource_outputs));

    var read_only_output = output_resource;
    read_only_output.access = .read_only;
    const bad_output_access = [_]backend_mod.ProgramIO{
        .external(4, 0, read_only_output, output.len * @sizeOf(f32)),
    };
    try std.testing.expect(!be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &bad_output_access));

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.uploadBindings(handle, runtime, &resource_persistent);
    be.executeConfiguredBindings(handle, runtime, false);

    var resource_output_values = [_]f32{ 0, 0, 0 };
    try readBufferF32(&compiled.gpu, external_output, resource_output_values[0..]);
    try std.testing.expectApproxEqAbs(@as(f32, 8), resource_output_values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9), resource_output_values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), resource_output_values[2], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 4), rt.call_count);
    try std.testing.expect(rt.backend_dispatch_count > 0);
    try std.testing.expectEqual(@as(u64, 2), rt.sync_count);
    try std.testing.expect(rt.program_command_shape.command_stencil_hash != 0);
}

test "wgpu backend configured executable hot path does not allocate" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const alloc = failing.allocator();
    var backend = WgpuBackend.init(alloc);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 2,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 2,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 2,
            },
        } },
        .{ .elementwise = .{ .op = .add, .dst = 4, .src0 = 3, .src1 = 2, .n = 2 } },
    };
    const buffer_sizes = [_]usize{ 2, 4, 2, 2, 2 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var weights = [_]f32{ 1, 0, 0, 1 };
    var bias = [_]f32{ 10, -1 };
    var input = [_]f32{ 2, 3 };
    var output = [_]f32{ -5, -5 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&bias), bias.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};
    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);

    failing.fail_index = failing.alloc_index;
    failing.resize_fail_index = failing.resize_index;
    const alloc_index = failing.alloc_index;
    const resize_index = failing.resize_index;

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.unchanged, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(0, 1)));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 12), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), output[1], 0.001);

    input = .{ 4, 5 };
    output = .{ -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[1], 0.001);

    output = .{ -9, -9 };
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 14), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), output[1], 0.001);

    try std.testing.expectEqual(alloc_index, failing.alloc_index);
    try std.testing.expectEqual(resize_index, failing.resize_index);
    try std.testing.expect(!failing.has_induced_failure);
}

test "wgpu backend executes standalone dense matmul program" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{
            .M = 2,
            .N = 2,
            .K = 3,
            .a_row_stride = 3,
            .a_col_stride = 1,
            .b_row_stride = 2,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 2,
        },
    } }};
    const buffer_sizes = [_]usize{ 6, 6, 4 };
    const uploads = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&a), a.len * @sizeOf(f32)),
        .host(1, 0, @ptrCast(&b), b.len * @sizeOf(f32)),
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);

    var output = [_]f32{ -1, -1, -1, -1 };
    const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};
    be.executeProgram(handle, &.{}, &outputs);
    try std.testing.expectApproxEqAbs(@as(f32, 58), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154), output[3], 0.001);
}

test "wgpu backend seeds bound runtimes with compile-time initial uploads" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 3,
    } }};
    const buffer_sizes = [_]usize{ 3, 3, 3 };
    var lhs = [_]f32{ 1, 2, 3 };
    var rhs = [_]f32{ 10, 20, 30 };
    const uploads = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&lhs), lhs.len * @sizeOf(f32)),
        .host(1, 0, @ptrCast(&rhs), rhs.len * @sizeOf(f32)),
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);

    lhs = [_]f32{ -100, -100, -100 };
    rhs = [_]f32{ -100, -100, -100 };

    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var output = [_]f32{ 0, 0, 0 };
    const outputs = [_]backend_mod.ProgramIO{
        .host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32)),
    };
    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &.{}, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);

    try std.testing.expectApproxEqAbs(@as(f32, 11), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 33), output[2], 0.001);
}

test "wgpu backend does not fuse away live matvec primary output" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .exp, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 2,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 2,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 2,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 2,
            },
        } },
        .{ .fused_elementwise = .{
            .steps = &fused_steps,
            .n = 2,
            .dst = 4,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 5,
            .src0 = 2,
            .src1 = 3,
            .n = 2,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 4, 2, 2, 2, 2 };
    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 0,
        0, 1,
    };
    var addend = [_]f32{ 10, 20 };
    const uploads = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32)),
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(3, 0, @ptrCast(&addend), addend.len * @sizeOf(f32)),
    };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &uploads,
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);

    var output = [_]f32{ 0, 0 };
    const outputs = [_]backend_mod.ProgramIO{
        .host(5, 0, @ptrCast(&output), output.len * @sizeOf(f32)),
    };
    be.executeProgram(handle, &.{}, &outputs);

    try std.testing.expectApproxEqAbs(@as(f32, 12), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 23), output[1], 0.001);
}

test "wgpu backend configured dense matmul supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{
            .M = 2,
            .N = 2,
            .K = 3,
            .a_row_stride = 3,
            .a_col_stride = 1,
            .b_row_stride = 2,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 2,
        },
    } }};
    const buffer_sizes = [_]usize{ 6, 6, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&b), b.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&a), a.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 58), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 139), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 154), output[3], 0.001);

    a = [_]f32{ 2, 0, 1, 1, 1, 1 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[3], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_a = try createBuffer(compiled.gpu.device, try bytesForElements(a.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_a);
    const external_b = try createBuffer(compiled.gpu.device, try bytesForElements(b.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_b);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_a_values = [_]f32{ 0, 1, 0, 1, 0, 1 };
    const resource_b_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try writeBufferF32(&compiled.gpu, external_a, &resource_a_values);
    try writeBufferF32(&compiled.gpu, external_b, &resource_b_values);

    var a_resource = try compiled.registerExternalBuffer(external_a, try bytesForElements(resource_a_values.len));
    a_resource.access = .read_only;
    var b_resource = try compiled.registerExternalBuffer(external_b, try bytesForElements(resource_b_values.len));
    b_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, b_resource, resource_b_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, a_resource, resource_a_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(2, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, 3), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), resource_output_host[3], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured qmatmul supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();
    try std.testing.expect(be.capabilities.qmatmul);
    try std.testing.expect(be.capabilities.runtime_qweights);

    const qdata = [_]i8{
        2,  -1, 3,
        4,  -2, 1,
        -3, 5,  2,
    };
    const scales = [_]f32{ 0.5, 0.25, 1.0 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 3,
        .cols = 3,
        .block_size = 4,
    }};
    const ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 2,
        .N = 3,
        .K = 3,
        .input_offset = 1,
        .input_row_stride = 4,
        .dst_offset = 1,
        .dst_row_stride = 4,
    } }};
    const buffer_sizes = [_]usize{ 9, 9 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 99, 1, 2, 3, 99, -1, 0.5, 4, 99 };
    var output = [_]f32{-7} ** 9;
    const expected = [_]f32{ -7, 2.75, 2.25, 8.0, -7, -3.0, 5.25, 6.625, -7 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &outputs);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 99, 2, 0, 1, 99, 1, 1, 1, 99 };
    output = [_]f32{-11} ** 9;
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -11), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -11), output[8], 0.001);

    var resource_output_host = [_]f32{0} ** 9;
    const resource_output_seed = [_]f32{-7} ** 9;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 99, 1, 2, 3, 99, -1, 0.5, 4, 99 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured qmatmul elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    var qdata = [_]i8{
        2,  -1, 3,
        4,  -2, 1,
        -3, 5,  2,
    };
    const scales = [_]f32{ 0.5, 0.25, 1.0 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 3,
        .cols = 3,
        .block_size = 4,
    }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 2,
            .N = 3,
            .K = 3,
            .input_row_stride = 3,
            .dst_row_stride = 3,
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 3,
            .src0 = 1,
            .src1 = 2,
            .src1_offset = 1,
            .n = 6,
        } },
    };
    const buffer_sizes = [_]usize{ 6, 6, 7, 6 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const dispatch_plan = inspectDispatchPlan(program);
    try std.testing.expect(dispatch_plan.supported);
    try std.testing.expectEqual(@as(u64, 2), dispatch_plan.covered_op_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.qmatmul_elementwise);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.quantizedProjectionCount());

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 1, 2, 3, -1, 0.5, 4 };
    var secondary = [_]f32{ 99, 1, -2, 3, 4, -5, 6 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1 };
    const expected = [_]f32{ 3.75, 0.25, 11.0, 1.0, 0.25, 12.625 };
    const persistent = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&secondary), secondary.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    try std.testing.expect(be.uploadQWeights(handle, runtime, &qweights));
    be.uploadBindings(handle, runtime, &persistent);
    qdata = [_]i8{0} ** 9;
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 2, 0, 1, 1, 1, 1 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[5], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_secondary = try createBuffer(compiled.gpu.device, try bytesForElements(secondary.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_secondary);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 1, 2, 3, -1, 0.5, 4 };
    const resource_secondary_values = [_]f32{ 99, 1, -2, 3, 4, -5, 6 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_secondary, &resource_secondary_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var secondary_resource = try compiled.registerExternalBuffer(external_secondary, try bytesForElements(resource_secondary_values.len));
    secondary_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(2, 0, secondary_resource, resource_secondary_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_dispatch_count);
}

test "wgpu backend runtime bindings own independent qweight state" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();
    try std.testing.expect(be.capabilities.runtime_qweights);

    const compile_qdata = [_]i8{ 1, 0, 0, 1 };
    const compile_scales = [_]f32{1};
    const compile_qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &compile_qdata,
        .scales = &compile_scales,
        .rows = 2,
        .cols = 2,
        .block_size = 4,
    }};
    const ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 2,
        .K = 2,
    } }};
    const buffer_sizes = [_]usize{ 2, 2 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &compile_qweights,
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime_a = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime_a);
    const runtime_b = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime_b);

    var input = [_]f32{ 2, 3 };
    var output_a = [_]f32{ -1, -1 };
    var output_b = [_]f32{ -1, -1 };
    const input_io = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const output_a_io = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output_a), output_a.len * @sizeOf(f32))};
    const output_b_io = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output_b), output_b.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime_a, &.{}, &input_io, &output_a_io));
    try std.testing.expect(be.configureBindings(handle, runtime_b, &.{}, &input_io, &output_b_io));

    const qweights_a = [_]backend_mod.QuantizedWeightUpload{compile_qweights[0]};
    var qdata_b = [_]i8{ 1, 2, 3, 4 };
    const scales_b = [_]f32{0.5};
    const qweights_b = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata_b,
        .scales = &scales_b,
        .rows = 2,
        .cols = 2,
        .block_size = 4,
    }};
    try std.testing.expect(be.uploadQWeights(handle, runtime_a, &qweights_a));
    try std.testing.expect(be.uploadQWeights(handle, runtime_b, &qweights_b));

    qdata_b = [_]i8{ 0, 0, 0, 0 };

    be.executeConfiguredBindings(handle, runtime_a, true);
    be.executeConfiguredBindings(handle, runtime_b, true);
    try std.testing.expectApproxEqAbs(@as(f32, 2), output_a[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), output_a[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), output_b[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), output_b[1], 0.001);

    const bad_qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &compile_qdata,
        .scales = &compile_scales,
        .rows = 3,
        .cols = 2,
        .block_size = 4,
    }};
    try std.testing.expect(!be.uploadQWeights(handle, runtime_a, &bad_qweights));
}

test "wgpu backend configured dense matvec slice store supports patch and resources" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 2,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 4,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 4,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 4,
            },
        } },
        .{ .slice_assign = .{
            .dst = 3,
            .src = 2,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 1,
            .src_row_stride = 1,
            .src_col_stride = 4,
            .patch_stride = 2,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 8, 4, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.uploadBindings(handle, runtime, &outputs);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(1, 1)));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 27), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[7], 0.001);

    var resource_output_host = [_]f32{0} ** 8;
    const resource_output_seed = [_]f32{-5} ** 8;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_weights_values = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(3, 1)));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 27), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 2), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 4), rt.backend_op_count);
}

test "wgpu backend configured qmatvec slice store supports patch and resources" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const qdata = [_]i8{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 2,
        .cols = 4,
        .block_size = 8,
    }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 4,
            .K = 2,
        } },
        .{ .slice_assign = .{
            .dst = 2,
            .src = 1,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 1,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 4, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &outputs);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(1, 1)));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 27), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[7], 0.001);

    var resource_output_host = [_]f32{0} ** 8;
    const resource_output_seed = [_]f32{-5} ** 8;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(2, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(3, 1)));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 27), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 2), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 4), rt.backend_op_count);
}

test "wgpu backend configured dense matvec rope slice store supports patch and resources" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 2,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 4,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 4,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 4,
            },
        } },
        .{ .rope = .{
            .dst = 4,
            .src = 2,
            .cos_sin = 3,
            .half_d = 1,
            .seq_len = 1,
            .src_off = 1,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 4,
            .cs_cs = 4,
        } },
        .{ .slice_assign = .{
            .dst = 5,
            .src = 4,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 8, 4, 4, 2, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    var cos_sin = [_]f32{ 0, 0, 1, 0 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(3, 0, @ptrCast(&cos_sin), cos_sin.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(5, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.uploadBindings(handle, runtime, &outputs);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(1, 1)));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -27), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[7], 0.001);

    var resource_output_host = [_]f32{0} ** 8;
    const resource_output_seed = [_]f32{-5} ** 8;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_cos_sin = try createBuffer(compiled.gpu.device, try bytesForElements(cos_sin.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_cos_sin);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_weights_values = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const resource_cos_sin_values = [_]f32{ 0, 0, 1, 0 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_cos_sin, &resource_cos_sin_values);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var cos_sin_resource = try compiled.registerExternalBuffer(external_cos_sin, try bytesForElements(resource_cos_sin_values.len));
    cos_sin_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(3, 0, cos_sin_resource, resource_cos_sin_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(5, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(3, 1)));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -27), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 2), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured qmatvec rope slice store supports patch and resources" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const qdata = [_]i8{
        1, 0, 0, 0, 1,  0,
        0, 1, 0, 0, 0,  1,
        0, 0, 1, 0, 1,  1,
        0, 0, 0, 1, -1, 1,
    };
    const scales = [_]f32{ 1, 1, 1, 1, 1, 1 };
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 4,
        .cols = 6,
        .block_size = 4,
    }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 1,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 6,
            .K = 4,
        } },
        .{ .rope = .{
            .dst = 2,
            .src = 1,
            .cos_sin = 3,
            .half_d = 3,
            .seq_len = 1,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 6,
            .cs_cs = 12,
        } },
        .{ .slice_assign = .{
            .dst = 4,
            .src = 2,
            .rows = 6,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 6,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 6,
            .patch_stride = 6,
        } },
    };
    const buffer_sizes = [_]usize{ 4, 6, 6, 12, 12 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 1, 2, 3, 4 };
    var cos_sin = [_]f32{ 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1 };
    var output = [_]f32{-1} ** 12;
    const expected = [_]f32{ -4, 2, -9, 1, 0, 3 };
    const persistent = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&cos_sin), cos_sin.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.uploadBindings(handle, runtime, &outputs);
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(1, 1)));
    be.executeConfiguredBindings(handle, runtime, true);
    for (output[0..6]) |got| try std.testing.expectApproxEqAbs(@as(f32, -1), got, 0.001);
    for (expected, output[6..12]) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var resource_output_host = [_]f32{0} ** 12;
    const resource_output_seed = [_]f32{-5} ** 12;
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_cos_sin = try createBuffer(compiled.gpu.device, try bytesForElements(cos_sin.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_cos_sin);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    try writeBufferF32(&compiled.gpu, external_input, &input);
    try writeBufferF32(&compiled.gpu, external_cos_sin, &cos_sin);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(input.len));
    input_resource.access = .read_only;
    var cos_sin_resource = try compiled.registerExternalBuffer(external_cos_sin, try bytesForElements(cos_sin.len));
    cos_sin_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(3, 0, cos_sin_resource, cos_sin.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, input.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime, try backend_mod.RuntimeWindow.init(0, 1)));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host[0..6]) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);
    for (resource_output_host[6..12]) |got| try std.testing.expectApproxEqAbs(@as(f32, -5), got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 2), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured qmatvec elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    var qdata = [_]i8{
        1, 2, 3,
        4, 5, 6,
    };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 2,
        .cols = 3,
        .block_size = 6,
    }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 2,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 3,
            .K = 2,
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 3,
            .src0 = 2,
            .src1 = 1,
            .n = 3,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var secondary = [_]f32{ 0.5, -1, 2 };
    var output = [_]f32{ -1, -1, -1 };
    const expected = [_]f32{ 7, -19, 48 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&secondary), secondary.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    try std.testing.expect(be.uploadQWeights(handle, runtime, &qweights));
    be.uploadBindings(handle, runtime, &persistent);
    qdata = [_]i8{0} ** 6;
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 4, 5 };
    output = [_]f32{ -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_secondary = try createBuffer(compiled.gpu.device, try bytesForElements(secondary.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_secondary);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_secondary_values = [_]f32{ 0.5, -1, 2 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_secondary, &resource_secondary_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var secondary_resource = try compiled.registerExternalBuffer(external_secondary, try bytesForElements(resource_secondary_values.len));
    secondary_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, secondary_resource, resource_secondary_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured qmatvec fused elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    var qdata = [_]i8{
        1, 2, 3,
        4, 5, 6,
    };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 2,
        .cols = 3,
        .block_size = 6,
    }};
    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .abs, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 2,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 3,
            .K = 2,
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 3,
            .dst = 3,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var addend = [_]f32{ 1, -2, 3 };
    var output = [_]f32{ -1, -1, -1 };
    const expected = [_]f32{ 210, 323, 648 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&addend), addend.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    try std.testing.expect(be.uploadQWeights(handle, runtime, &qweights));
    be.uploadBindings(handle, runtime, &persistent);
    qdata = [_]i8{0} ** 6;
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 4, 5 };
    output = [_]f32{ -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_addend = try createBuffer(compiled.gpu.device, try bytesForElements(addend.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_addend);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_addend_values = [_]f32{ 1, -2, 3 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_addend, &resource_addend_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var addend_resource = try compiled.registerExternalBuffer(external_addend, try bytesForElements(resource_addend_values.len));
    addend_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, addend_resource, resource_addend_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured dense matvec elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 3,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 3,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 3,
            },
        } },
        .{ .elementwise = .{
            .op = .mul,
            .dst = 4,
            .src0 = 3,
            .src1 = 2,
            .n = 3,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 6, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var secondary = [_]f32{ 0.5, -1, 2 };
    var output = [_]f32{ -1, -1, -1 };
    const expected = [_]f32{ 7, -19, 48 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&secondary), secondary.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 4, 5 };
    output = [_]f32{ -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_secondary = try createBuffer(compiled.gpu.device, try bytesForElements(secondary.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_secondary);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_weights_values = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const resource_secondary_values = [_]f32{ 0.5, -1, 2 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_secondary, &resource_secondary_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var secondary_resource = try compiled.registerExternalBuffer(external_secondary, try bytesForElements(resource_secondary_values.len));
    secondary_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, secondary_resource, resource_secondary_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured dense matvec fused elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .abs, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 3, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 1,
                .N = 3,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 3,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 3,
            },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 3,
            .dst = 4,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 6, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var weights = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var addend = [_]f32{ 1, -2, 3 };
    var output = [_]f32{ -1, -1, -1 };
    const expected = [_]f32{ 210, 323, 648 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&addend), addend.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 4, 5 };
    output = [_]f32{ -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_addend = try createBuffer(compiled.gpu.device, try bytesForElements(addend.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_addend);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_weights_values = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const resource_addend_values = [_]f32{ 1, -2, 3 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_addend, &resource_addend_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var addend_resource = try compiled.registerExternalBuffer(external_addend, try bytesForElements(resource_addend_values.len));
    addend_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, addend_resource, resource_addend_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured dense matmul elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 2,
                .N = 3,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 3,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 3,
            },
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 4,
            .src0 = 3,
            .src1 = 2,
            .n = 6,
        } },
    };
    const buffer_sizes = [_]usize{ 4, 6, 6, 6, 6 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const dispatch_plan = inspectDispatchPlan(program);
    try std.testing.expect(dispatch_plan.supported);
    try std.testing.expectEqual(@as(u64, 2), dispatch_plan.covered_op_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.matmul_elementwise);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.projectionCount());

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3, 4, 5 };
    var weights = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var secondary = [_]f32{ 1, -2, 3, 4, -5, 6 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1 };
    const expected = [_]f32{ 15, 17, 27, 28, 28, 48 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&secondary), secondary.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 5, 6, 7, 8 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[5], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_secondary = try createBuffer(compiled.gpu.device, try bytesForElements(secondary.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_secondary);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3, 4, 5 };
    const resource_weights_values = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const resource_secondary_values = [_]f32{ 1, -2, 3, 4, -5, 6 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_secondary, &resource_secondary_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var secondary_resource = try compiled.registerExternalBuffer(external_secondary, try bytesForElements(resource_secondary_values.len));
    secondary_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, secondary_resource, resource_secondary_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_dispatch_count);
}

test "wgpu backend configured dense matmul fused elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .neg, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .abs, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .matmul = .{
            .dst = 3,
            .a = 0,
            .b = 1,
            .geom = .{
                .M = 2,
                .N = 3,
                .K = 2,
                .a_row_stride = 2,
                .a_col_stride = 1,
                .b_row_stride = 3,
                .b_col_stride = 1,
                .a_offset = 0,
                .b_offset = 0,
                .dst_offset = 0,
                .dst_row_stride = 3,
            },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 6,
            .dst = 5,
            .src = 3,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const buffer_sizes = [_]usize{ 4, 6, 6, 6, 6, 6 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const dispatch_plan = inspectDispatchPlan(program);
    try std.testing.expect(dispatch_plan.supported);
    try std.testing.expectEqual(@as(u64, 2), dispatch_plan.covered_op_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.dispatch_count);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.matmul_fused_elementwise);
    try std.testing.expectEqual(@as(u64, 1), dispatch_plan.family_counts.projectionCount());

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3, 4, 5 };
    var weights = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    var addend = [_]f32{ 1, -2, 3, 4, -5, 6 };
    var scale = [_]f32{ 10, 20, 30, 1, 2, 3 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1 };
    const expected = [_]f32{ 150, 340, 810, 28, 56, 144 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&weights), weights.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&addend), addend.len * @sizeOf(f32)),
        .host(4, 0, @ptrCast(&scale), scale.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(5, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 5, 6, 7, 8 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[5], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_weights = try createBuffer(compiled.gpu.device, try bytesForElements(weights.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_weights);
    const external_addend = try createBuffer(compiled.gpu.device, try bytesForElements(addend.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_addend);
    const external_scale = try createBuffer(compiled.gpu.device, try bytesForElements(scale.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_scale);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3, 4, 5 };
    const resource_weights_values = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const resource_addend_values = [_]f32{ 1, -2, 3, 4, -5, 6 };
    const resource_scale_values = [_]f32{ 10, 20, 30, 1, 2, 3 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_weights, &resource_weights_values);
    try writeBufferF32(&compiled.gpu, external_addend, &resource_addend_values);
    try writeBufferF32(&compiled.gpu, external_scale, &resource_scale_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var weights_resource = try compiled.registerExternalBuffer(external_weights, try bytesForElements(resource_weights_values.len));
    weights_resource.access = .read_only;
    var addend_resource = try compiled.registerExternalBuffer(external_addend, try bytesForElements(resource_addend_values.len));
    addend_resource.access = .read_only;
    var scale_resource = try compiled.registerExternalBuffer(external_scale, try bytesForElements(resource_scale_values.len));
    scale_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, weights_resource, resource_weights_values.len * @sizeOf(f32)),
        .external(2, 0, addend_resource, resource_addend_values.len * @sizeOf(f32)),
        .external(4, 0, scale_resource, resource_scale_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(5, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_dispatch_count);
}

test "wgpu backend configured qlinear supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    var qdata = [_]i8{
        1, 2, 3,
        4, 5, 6,
    };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 2,
        .cols = 3,
        .block_size = 6,
    }};
    const ops = [_]backend_mod.DeviceOp{
        .{ .qmatmul = .{
            .dst = 2,
            .input = 0,
            .weight_idx = 0,
            .M = 1,
            .N = 3,
            .K = 2,
        } },
        .{ .elementwise = .{
            .op = .add,
            .dst = 3,
            .src0 = 2,
            .src1 = 1,
            .n = 3,
        } },
    };
    const buffer_sizes = [_]usize{ 2, 3, 3, 3 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
        .qweights = &qweights,
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var input = [_]f32{ 2, 3 };
    var bias = [_]f32{ 0.5, -1, 2 };
    var output = [_]f32{ -1, -1, -1 };
    const expected = [_]f32{ 14.5, 18, 26 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&bias), bias.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&input), input.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    try std.testing.expect(be.uploadQWeights(handle, runtime, &qweights));
    be.uploadBindings(handle, runtime, &persistent);
    qdata = [_]i8{0} ** 6;
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    input = [_]f32{ 4, 5 };
    output = [_]f32{ -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_input = try createBuffer(compiled.gpu.device, try bytesForElements(input.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_input);
    const external_bias = try createBuffer(compiled.gpu.device, try bytesForElements(bias.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_bias);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_input_values = [_]f32{ 2, 3 };
    const resource_bias_values = [_]f32{ 0.5, -1, 2 };
    try writeBufferF32(&compiled.gpu, external_input, &resource_input_values);
    try writeBufferF32(&compiled.gpu, external_bias, &resource_bias_values);

    var input_resource = try compiled.registerExternalBuffer(external_input, try bytesForElements(resource_input_values.len));
    input_resource.access = .read_only;
    var bias_resource = try compiled.registerExternalBuffer(external_bias, try bytesForElements(resource_bias_values.len));
    bias_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, bias_resource, resource_bias_values.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, input_resource, resource_input_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const buffer_sizes = [_]usize{ 4, 4, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src0 = [_]f32{ 1, 2, 3, 4 };
    var src1 = [_]f32{ 10, 20, 30, 40 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&src1), src1.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src0), src0.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 11), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 33), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44), output[3], 0.001);

    src0 = [_]f32{ 5, 6, 7, 8 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[3], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src0 = try createBuffer(compiled.gpu.device, try bytesForElements(src0.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src0);
    const external_src1 = try createBuffer(compiled.gpu.device, try bytesForElements(src1.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src1);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src0_values = [_]f32{ -1, 0, 1, 2 };
    const resource_src1_values = [_]f32{ 2, 3, 4, 5 };
    try writeBufferF32(&compiled.gpu, external_src0, &resource_src0_values);
    try writeBufferF32(&compiled.gpu, external_src1, &resource_src1_values);

    var src0_resource = try compiled.registerExternalBuffer(external_src0, try bytesForElements(resource_src0_values.len));
    src0_resource.access = .read_only;
    var src1_resource = try compiled.registerExternalBuffer(external_src1, try bytesForElements(resource_src1_values.len));
    src1_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{
        .external(0, 0, src0_resource, resource_src0_values.len * @sizeOf(f32)),
        .external(1, 0, src1_resource, resource_src1_values.len * @sizeOf(f32)),
    };
    const resource_outputs = [_]backend_mod.ProgramIO{.external(2, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, 1), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), resource_output_host[3], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured sgn and step elementwise semantics" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const sgn_ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .sgn,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const step_ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .step,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .sgn, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .step, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const fused_ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &fused_steps,
        .n = 4,
        .dst = 2,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};
    const buffer_sizes = [_]usize{ 4, 4, 4 };
    var src = [_]f32{ -2, 0, 3, -0.5 };
    var unused = [_]f32{ 0, 0, 0, 0 };
    const inputs = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32)),
        .host(1, 0, @ptrCast(&unused), unused.len * @sizeOf(f32)),
    };

    const programs = [_]struct {
        ops: []const backend_mod.DeviceOp,
        expected: [4]f32,
    }{
        .{ .ops = &sgn_ops, .expected = .{ -1, 0, 1, -1 } },
        .{ .ops = &step_ops, .expected = .{ 0, 0, 1, 0 } },
        .{ .ops = &fused_ops, .expected = .{ 0, 0, 1, 0 } },
    };

    for (programs) |case| {
        const program = backend_mod.DeviceProgram{
            .ops = case.ops,
            .n_buffers = buffer_sizes.len,
            .buffer_sizes = &buffer_sizes,
            .initial_uploads = &.{},
        };
        try std.testing.expect(be.supportsProgram(program));
        const handle = be.compileProgram(program) orelse return error.SkipZigTest;
        defer be.freeProgram(handle);
        const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
        defer be.freeBindings(handle, runtime);

        var output = [_]f32{ -1, -1, -1, -1 };
        const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};
        try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
        be.executeConfiguredBindings(handle, runtime, true);
        for (case.expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);
    }
}

test "wgpu backend configured fused elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &steps,
        .n = 4,
        .dst = 3,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};
    const buffer_sizes = [_]usize{ 4, 4, 4, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ -2, -1, 0.5, 2 };
    var addend = [_]f32{ 10, 20, 30, 40 };
    var scale = [_]f32{ 2, 3, 4, 5 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const expected = [_]f32{ 20, 60, 122, 210 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&addend), addend.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&scale), scale.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(3, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    src = [_]f32{ 1, 2, 3, 4 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    for (output) |got| try std.testing.expectApproxEqAbs(@as(f32, -7), got, 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_addend = try createBuffer(compiled.gpu.device, try bytesForElements(addend.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_addend);
    const external_scale = try createBuffer(compiled.gpu.device, try bytesForElements(scale.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_scale);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ -2, -1, 0.5, 2 };
    const resource_addend_values = [_]f32{ 10, 20, 30, 40 };
    const resource_scale_values = [_]f32{ 2, 3, 4, 5 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);
    try writeBufferF32(&compiled.gpu, external_addend, &resource_addend_values);
    try writeBufferF32(&compiled.gpu, external_scale, &resource_scale_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var addend_resource = try compiled.registerExternalBuffer(external_addend, try bytesForElements(resource_addend_values.len));
    addend_resource.access = .read_only;
    var scale_resource = try compiled.registerExternalBuffer(external_scale, try bytesForElements(resource_scale_values.len));
    scale_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, addend_resource, resource_addend_values.len * @sizeOf(f32)),
        .external(2, 0, scale_resource, resource_scale_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(3, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_op_count);
}

test "wgpu backend executes bounded multi-dispatch elementwise chain" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{
        .{ .elementwise = .{ .op = .add, .dst = 2, .src0 = 0, .src1 = 1, .n = 4 } },
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 2, .src1 = 3, .n = 4 } },
    };
    const buffer_sizes = [_]usize{ 4, 4, 4, 4, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 10, 20, 30, 40 };
    var scale = [_]f32{ 2, 3, 4, 5 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const expected = [_]f32{ 22, 66, 132, 220 };
    const inputs = [_]backend_mod.ProgramIO{
        .host(0, 0, @ptrCast(&a), a.len * @sizeOf(f32)),
        .host(1, 0, @ptrCast(&b), b.len * @sizeOf(f32)),
        .host(3, 0, @ptrCast(&scale), scale.len * @sizeOf(f32)),
    };
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    a = [_]f32{ 5, 6, 7, 8 };
    b = [_]f32{ 1, 1, 1, 1 };
    scale = [_]f32{ 10, 20, 30, 40 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    for (output) |got| try std.testing.expectApproxEqAbs(@as(f32, -7), got, 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_a = try createBuffer(compiled.gpu.device, try bytesForElements(a.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_a);
    const external_b = try createBuffer(compiled.gpu.device, try bytesForElements(b.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_b);
    const external_scale = try createBuffer(compiled.gpu.device, try bytesForElements(scale.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_scale);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_a_values = [_]f32{ 2, 4, 6, 8 };
    const resource_b_values = [_]f32{ 1, 3, 5, 7 };
    const resource_scale_values = [_]f32{ 1, 2, 3, 4 };
    const resource_expected = [_]f32{ 3, 14, 33, 60 };
    try writeBufferF32(&compiled.gpu, external_a, &resource_a_values);
    try writeBufferF32(&compiled.gpu, external_b, &resource_b_values);
    try writeBufferF32(&compiled.gpu, external_scale, &resource_scale_values);

    var a_resource = try compiled.registerExternalBuffer(external_a, try bytesForElements(resource_a_values.len));
    a_resource.access = .read_only;
    var b_resource = try compiled.registerExternalBuffer(external_b, try bytesForElements(resource_b_values.len));
    b_resource.access = .read_only;
    var scale_resource = try compiled.registerExternalBuffer(external_scale, try bytesForElements(resource_scale_values.len));
    scale_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{
        .external(0, 0, a_resource, resource_a_values.len * @sizeOf(f32)),
        .external(1, 0, b_resource, resource_b_values.len * @sizeOf(f32)),
        .external(3, 0, scale_resource, resource_scale_values.len * @sizeOf(f32)),
    };
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (resource_expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_dispatch_count);
}

test "wgpu backend configured repeat supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .repeat = .{
        .dst = 1,
        .src = 0,
        .n = 8,
        .src_ne = .{ 2, 1, 1, 1 },
        .dst_ne = .{ 2, 4, 1, 1 },
        .src_strides = .{ 1, 2, 2, 2 },
        .dst_strides = .{ 1, 2, 8, 8 },
    } }};
    const buffer_sizes = [_]usize{ 2, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 10, 20 };
    var output = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const expected = [_]f32{ 10, 20, 10, 20, 10, 20, 10, 20 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    src = [_]f32{ 30, 40 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    for (output) |got| try std.testing.expectApproxEqAbs(@as(f32, -7), got, 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 5, 9 };
    const resource_expected = [_]f32{ 5, 9, 5, 9, 5, 9, 5, 9 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (resource_expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 3), rt.backend_op_count);
}

test "wgpu backend configured repeat fused elementwise supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = true, .secondary_buf = 3, .secondary_offset = 0 },
    };
    const ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &steps,
            .n = 4,
            .dst = 4,
            .src = 0,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const buffer_sizes = [_]usize{ 4, 1, 4, 4, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 1, 3, 7, 15 };
    var one = [_]f32{1};
    var scale = [_]f32{ 2, 3, 4, 5 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const expected = [_]f32{ 1, 0.75, 0.5, 0.3125 };
    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&one), one.len * @sizeOf(f32)),
        .host(3, 0, @ptrCast(&scale), scale.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    src = [_]f32{ 2, 4, 8, 16 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    for (output) |got| try std.testing.expectApproxEqAbs(@as(f32, -7), got, 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_one = try createBuffer(compiled.gpu.device, try bytesForElements(one.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_one);
    const external_scale = try createBuffer(compiled.gpu.device, try bytesForElements(scale.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_scale);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 1, 3, 7, 15 };
    const resource_one_values = [_]f32{1};
    const resource_scale_values = [_]f32{ 2, 3, 4, 5 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);
    try writeBufferF32(&compiled.gpu, external_one, &resource_one_values);
    try writeBufferF32(&compiled.gpu, external_scale, &resource_scale_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var one_resource = try compiled.registerExternalBuffer(external_one, try bytesForElements(resource_one_values.len));
    one_resource.access = .read_only;
    var scale_resource = try compiled.registerExternalBuffer(external_scale, try bytesForElements(resource_scale_values.len));
    scale_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, one_resource, resource_one_values.len * @sizeOf(f32)),
        .external(3, 0, scale_resource, resource_scale_values.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
    try std.testing.expectEqual(@as(u64, 6), rt.backend_op_count);
}

test "wgpu backend configured layernorm supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .layernorm = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 4,
        .eps = 0.0,
    } }};
    const buffer_sizes = [_]usize{ 4, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    try std.testing.expect(be.supportsProgram(program));
    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 1, 2, 3, 4 };
    var output = [_]f32{ -1, -1, -1, -1 };
    const expected = [_]f32{ -1.3416408, -0.4472136, 0.4472136, 1.3416408 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    src = [_]f32{ 4, 3, 2, 1 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    for (output) |got| try std.testing.expectApproxEqAbs(@as(f32, -7), got, 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 2, 4, 6, 8 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured rmsnorm supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .rmsnorm = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 4,
        .eps = 0.0,
    } }};
    const buffer_sizes = [_]usize{ 8, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 1, 1, 1, 1, 2, 0, 0, 0 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 1), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), output[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), output[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), output[7], 0.001);

    src = [_]f32{ 0, 2, 0, 0, 2, 2, 2, 2 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[7], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 3, 0, 0, 0, 0, 2, 0, 0 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, 2), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), resource_output_host[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured softmax supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .softmax = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 4,
    } }};
    const buffer_sizes = [_]usize{ 8, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 0, 0, 0, 0, 1, 2, 3, 4 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0320586), output[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0871443), output[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2368828), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6439143), output[7], 0.001);

    src = [_]f32{ 4, 3, 2, 1, 0, 0, 0, 0 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[7], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 2, 2, 2, 2, -1, 0, 1, 2 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), resource_output_host[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0320586), resource_output_host[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0871443), resource_output_host[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2368828), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6439143), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend executes DeviceInference log-softmax lowering" {
    if (!options.use_wgpu) return error.SkipZigTest;

    const DeviceF32 = DeviceInference(f32);
    const GraphF32 = ComputeGraph(f32);
    const TensorF32 = Tensor(f32);

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();

    var graph = GraphF32.init(std.testing.allocator);
    defer graph.deinit();
    const alloc = graph.allocator();

    const input = try TensorF32.init(alloc, &.{ 4, 2 });
    const output = input.logSoftmaxDim(0);
    try graph.infer(output);

    var program = try DeviceF32.Program.compile(.{
        .graph = &graph,
        .be = backend.backend(),
        .alloc = std.testing.allocator,
        .input_tensors = &.{input},
        .output_tensors = &.{output},
    });
    defer program.deinit();

    const inspection = program.inspect();
    try std.testing.expectEqual(@as(u64, 1), inspection.op_count);
    try std.testing.expect(inspection.execution_supported);
    try std.testing.expectEqual(@as(u32, 1), inspection.command_shape.command_count);

    var input_values = [_]f32{
        1, 2,  3,  4,
        0, -1, -2, -3,
    };
    var output_values = [_]f32{0} ** input_values.len;
    const input_binding = DeviceF32.TensorBinding.host(input, input_values[0..].ptr, input_values.len);

    var session = try program.bind(.{
        .input_tensors = &.{input},
        .input_bindings = &.{input_binding},
        .output_tensor = output,
        .output_host_buf = output_values[0..].ptr,
        .output_len = output_values.len,
    });
    defer session.deinit();

    try program.executeStep(&session, .{ .window = try backend_mod.RuntimeWindow.init(0, 0) });

    const denom_a = @log(@exp(@as(f32, 1)) + @exp(@as(f32, 2)) + @exp(@as(f32, 3)) + @exp(@as(f32, 4)));
    const denom_b = @log(@exp(@as(f32, 0)) + @exp(@as(f32, -1)) + @exp(@as(f32, -2)) + @exp(@as(f32, -3)));
    const expected = [_]f32{
        1 - denom_a,
        2 - denom_a,
        3 - denom_a,
        4 - denom_a,
        0 - denom_b,
        -1 - denom_b,
        -2 - denom_b,
        -3 - denom_b,
    };
    for (expected, output_values) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);
}

test "wgpu backend configured reduce supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .reduce = .{
        .op = .sum,
        .dst = 1,
        .src = 0,
        .n_out = 2,
        .reduce_size = 4,
    } }};
    const buffer_sizes = [_]usize{ 8, 2 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 1, 2, 3, 4, -1, -2, -3, -4 };
    var output = [_]f32{ -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, 10), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -10), output[1], 0.001);

    src = [_]f32{ 2, 2, 2, 2, 1, 1, 1, 1 };
    output = [_]f32{ -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[1], 0.001);

    var resource_output_host = [_]f32{ 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 4, 4, 4, 4, -2, 0, 2, 4 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, 16), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), resource_output_host[1], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);

    var max_src = [_]f32{ 1, -2, 3, 0, -1, -5, -3, -4 };
    const max_ops = [_]backend_mod.DeviceOp{.{ .reduce = .{
        .op = .max,
        .dst = 1,
        .src = 0,
        .n_out = 2,
        .reduce_size = 4,
    } }};
    const max_uploads = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&max_src), max_src.len * @sizeOf(f32))};
    const max_program = backend_mod.DeviceProgram{
        .ops = &max_ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &max_uploads,
    };
    const max_handle = be.compileProgram(max_program) orelse return error.SkipZigTest;
    defer be.freeProgram(max_handle);

    var max_output = [_]f32{ 0, 0 };
    const max_outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&max_output), max_output.len * @sizeOf(f32))};
    be.executeProgram(max_handle, &.{}, &max_outputs);
    try std.testing.expectApproxEqAbs(@as(f32, 3), max_output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), max_output[1], 0.001);

    var min_src = [_]f32{ 1, -2, 3, 0, -1, -5, -3, -4 };
    const min_ops = [_]backend_mod.DeviceOp{.{ .reduce = .{
        .op = .min,
        .dst = 1,
        .src = 0,
        .n_out = 2,
        .reduce_size = 4,
    } }};
    const min_uploads = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&min_src), min_src.len * @sizeOf(f32))};
    const min_program = backend_mod.DeviceProgram{
        .ops = &min_ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &min_uploads,
    };
    const min_handle = be.compileProgram(min_program) orelse return error.SkipZigTest;
    defer be.freeProgram(min_handle);

    var min_output = [_]f32{ 0, 0 };
    const min_outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&min_output), min_output.len * @sizeOf(f32))};
    be.executeProgram(min_handle, &.{}, &min_outputs);
    try std.testing.expectApproxEqAbs(@as(f32, -2), min_output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -5), min_output[1], 0.001);
}

test "wgpu backend configured rope supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .rope = .{
        .dst = 2,
        .src = 0,
        .cos_sin = 1,
        .half_d = 2,
        .seq_len = 2,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 8,
    } }};
    const buffer_sizes = [_]usize{ 8, 16, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var cos_sin = [_]f32{ 0.5, 0.25, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const persistent = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&cos_sin), cos_sin.len * @sizeOf(f32))};
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(2, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7.5), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), output[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), output[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), output[7], 0.001);

    src = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[7], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_cos_sin = try createBuffer(compiled.gpu.device, try bytesForElements(cos_sin.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_cos_sin);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);
    try writeBufferF32(&compiled.gpu, external_cos_sin, &cos_sin);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var cos_sin_resource = try compiled.registerExternalBuffer(external_cos_sin, try bytesForElements(cos_sin.len));
    cos_sin_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{.external(1, 0, cos_sin_resource, cos_sin.len * @sizeOf(f32))};
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(2, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7.5), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), resource_output_host[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), resource_output_host[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), resource_output_host[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), resource_output_host[7], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured slice assign supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 3,
        .dst_base_offset = 0,
        .dst_offset = 1,
        .dst_row_stride = 1,
        .dst_col_stride = 3,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 3,
    } }};
    const buffer_sizes = [_]usize{ 6, 10 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var src = [_]f32{ 10, 20, 30, 40, 50, 60 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &outputs);
    be.executeConfiguredBindings(handle, runtime, true);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30), output[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 40), output[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50), output[7], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 60), output[8], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output[9], 0.001);

    src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    output = [_]f32{ -7, -7, -7, -7, -7, -7, -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[9], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const resource_output_seed = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_src = try createBuffer(compiled.gpu.device, try bytesForElements(src.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_src);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_output);

    const resource_src_values = [_]f32{ 10, 20, 30, 40, 50, 60 };
    try writeBufferF32(&compiled.gpu, external_src, &resource_src_values);
    try writeBufferF32(&compiled.gpu, external_output, &resource_output_seed);

    var src_resource = try compiled.registerExternalBuffer(external_src, try bytesForElements(resource_src_values.len));
    src_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, src_resource, resource_src_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(1, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &.{}, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    try std.testing.expectApproxEqAbs(@as(f32, -1), resource_output_host[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), resource_output_host[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20), resource_output_host[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), resource_output_host[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30), resource_output_host[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 40), resource_output_host[5], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), resource_output_host[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 50), resource_output_host[7], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 60), resource_output_host[8], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), resource_output_host[9], 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend runtime patch params are binding-local" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 1,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 1,
        .patch_stride = 2,
    } }};
    const buffer_sizes = [_]usize{ 2, 8 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime_a = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime_a);
    const runtime_b = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime_b);

    var src = [_]f32{ 10, 20 };
    var output_a = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    var output_b = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    const input_io = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&src), src.len * @sizeOf(f32))};
    const output_a_io = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output_a), output_a.len * @sizeOf(f32))};
    const output_b_io = [_]backend_mod.ProgramIO{.host(1, 0, @ptrCast(&output_b), output_b.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime_a, &.{}, &input_io, &output_a_io));
    try std.testing.expect(be.configureBindings(handle, runtime_b, &.{}, &input_io, &output_b_io));
    be.uploadBindings(handle, runtime_a, &output_a_io);
    be.uploadBindings(handle, runtime_b, &output_b_io);

    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime_a, try backend_mod.RuntimeWindow.init(1, 1)));
    try std.testing.expectEqual(backend_mod.RuntimePatchStatus.changed, be.patchRuntimeBindings(handle, runtime_b, try backend_mod.RuntimeWindow.init(3, 1)));

    be.executeConfiguredBindings(handle, runtime_a, true);
    be.executeConfiguredBindings(handle, runtime_b, true);

    try std.testing.expectApproxEqAbs(@as(f32, -1), output_a[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output_a[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), output_a[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20), output_a[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output_a[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output_a[7], 0.001);

    try std.testing.expectApproxEqAbs(@as(f32, -1), output_b[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1), output_b[3], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), output_b[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20), output_b[7], 0.001);
}

test "wgpu backend configured attention supports host and resource bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = 2,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 2,
        .dst_rs = 1,
        .dst_cs = 4,
    } }};
    const buffer_sizes = [_]usize{ 4, 8, 8, 2, 4 };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var q = [_]f32{ 0.2, -0.4, 0.6, -0.8 };
    var k = [_]f32{
        0.1,  0.3, -0.5, 0.7,
        -0.2, 0.4, 0.8,  -0.6,
    };
    var v = [_]f32{
        1.0,  2.0,  -1.0, 0.5,
        -0.5, 0.25, 1.5,  -2.0,
    };
    var mask = [_]f32{ 0, 0 };
    var output = [_]f32{ 0, 0, 0, 0 };

    const s0 = 0.5 * (q[0] * k[0] + q[1] * k[1] + q[2] * k[2] + q[3] * k[3]);
    const s1 = 0.5 * (q[0] * k[4] + q[1] * k[5] + q[2] * k[6] + q[3] * k[7]);
    const m = @max(s0, s1);
    const e0 = @exp(s0 - m);
    const e1 = @exp(s1 - m);
    const inv = 1.0 / (e0 + e1);
    const w0 = e0 * inv;
    const w1 = e1 * inv;
    const expected = [_]f32{
        w0 * v[0] + w1 * v[4],
        w0 * v[1] + w1 * v[5],
        w0 * v[2] + w1 * v[6],
        w0 * v[3] + w1 * v[7],
    };

    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(&k), k.len * @sizeOf(f32)),
        .host(2, 0, @ptrCast(&v), v.len * @sizeOf(f32)),
        .host(3, 0, @ptrCast(&mask), mask.len * @sizeOf(f32)),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&q), q.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    q = [_]f32{ 1, 1, 1, 1 };
    output = [_]f32{ -7, -7, -7, -7 };
    be.executeConfiguredBindings(handle, runtime, false);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -7), output[3], 0.001);

    var resource_output_host = [_]f32{ 0, 0, 0, 0 };
    const compiled: *CompiledProgram = @ptrCast(@alignCast(handle));
    const external_q = try createBuffer(compiled.gpu.device, try bytesForElements(q.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_q);
    const external_k = try createBuffer(compiled.gpu.device, try bytesForElements(k.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_k);
    const external_v = try createBuffer(compiled.gpu.device, try bytesForElements(v.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_v);
    const external_mask = try createBuffer(compiled.gpu.device, try bytesForElements(mask.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopyDst);
    defer c.wgpuBufferRelease(external_mask);
    const external_output = try createBuffer(compiled.gpu.device, try bytesForElements(resource_output_host.len), c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc);
    defer c.wgpuBufferRelease(external_output);

    const resource_q_values = [_]f32{ 0.2, -0.4, 0.6, -0.8 };
    try writeBufferF32(&compiled.gpu, external_q, &resource_q_values);
    try writeBufferF32(&compiled.gpu, external_k, &k);
    try writeBufferF32(&compiled.gpu, external_v, &v);
    try writeBufferF32(&compiled.gpu, external_mask, &mask);

    var q_resource = try compiled.registerExternalBuffer(external_q, try bytesForElements(resource_q_values.len));
    q_resource.access = .read_only;
    var k_resource = try compiled.registerExternalBuffer(external_k, try bytesForElements(k.len));
    k_resource.access = .read_only;
    var v_resource = try compiled.registerExternalBuffer(external_v, try bytesForElements(v.len));
    v_resource.access = .read_only;
    var mask_resource = try compiled.registerExternalBuffer(external_mask, try bytesForElements(mask.len));
    mask_resource.access = .read_only;
    var output_resource = try compiled.registerExternalBuffer(external_output, try bytesForElements(resource_output_host.len));
    output_resource.access = .write_only;

    const resource_persistent = [_]backend_mod.ProgramIO{
        .external(1, 0, k_resource, k.len * @sizeOf(f32)),
        .external(2, 0, v_resource, v.len * @sizeOf(f32)),
        .external(3, 0, mask_resource, mask.len * @sizeOf(f32)),
    };
    const resource_inputs = [_]backend_mod.ProgramIO{.external(0, 0, q_resource, resource_q_values.len * @sizeOf(f32))};
    const resource_outputs = [_]backend_mod.ProgramIO{.external(4, 0, output_resource, resource_output_host.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &resource_persistent, &resource_inputs, &resource_outputs));
    be.executeConfiguredBindings(handle, runtime, false);
    try readBufferF32(&compiled.gpu, external_output, &resource_output_host);
    for (expected, resource_output_host) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);

    var rt = profile_mod.RuntimeProfile{};
    be.addRuntimeBindingsProfileTo(handle, runtime, &rt);
    try std.testing.expectEqual(@as(u32, 3), rt.call_count);
    try std.testing.expectEqual(@as(u64, 1), rt.sync_count);
}

test "wgpu backend configured long attention supports host bindings" {
    if (!options.use_wgpu) return error.SkipZigTest;

    const seq_kv = 384;
    const d_head = 4;
    const scale: f32 = 0.5;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = d_head,
        .seq_q = 1,
        .seq_kv = seq_kv,
        .scale = scale,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = d_head,
        .k_rs = 1,
        .k_cs = d_head,
        .v_rs = 1,
        .v_cs = d_head,
        .mask_rs = 1,
        .mask_cs = seq_kv,
        .dst_rs = 1,
        .dst_cs = d_head,
    } }};
    const buffer_sizes = [_]usize{ d_head, seq_kv * d_head, seq_kv * d_head, seq_kv, d_head };
    const program = backend_mod.DeviceProgram{
        .ops = &ops,
        .n_buffers = buffer_sizes.len,
        .buffer_sizes = &buffer_sizes,
        .initial_uploads = &.{},
    };

    const handle = be.compileProgram(program) orelse return error.SkipZigTest;
    defer be.freeProgram(handle);
    const runtime = be.bindProgram(handle) orelse return error.SkipZigTest;
    defer be.freeBindings(handle, runtime);

    var q = [_]f32{ 0.2, -0.4, 0.6, -0.8 };
    var k = try std.testing.allocator.alloc(f32, seq_kv * d_head);
    defer std.testing.allocator.free(k);
    var v = try std.testing.allocator.alloc(f32, seq_kv * d_head);
    defer std.testing.allocator.free(v);
    var mask = try std.testing.allocator.alloc(f32, seq_kv);
    defer std.testing.allocator.free(mask);
    var output = [_]f32{ 0, 0, 0, 0 };

    for (0..seq_kv) |s| {
        const centered_s: i32 = @as(i32, @intCast(s % 23)) - 11;
        mask[s] = if (s % 17 == 0) -0.125 else 0.0;
        for (0..d_head) |r| {
            const centered_r: i32 = @as(i32, @intCast(r)) - 1;
            k[s * d_head + r] = @as(f32, @floatFromInt(centered_s + centered_r)) * 0.01;
            v[s * d_head + r] = @as(f32, @floatFromInt((@as(i32, @intCast(s % 29)) - 14) + centered_r * 2)) * 0.02;
        }
    }

    var max_score = -std.math.floatMax(f32);
    for (0..seq_kv) |s| {
        var dot: f32 = 0;
        for (0..d_head) |r| dot += q[r] * k[s * d_head + r];
        max_score = @max(max_score, dot * scale + mask[s]);
    }
    var denom: f32 = 0;
    var expected = [_]f32{ 0, 0, 0, 0 };
    for (0..seq_kv) |s| {
        var dot: f32 = 0;
        for (0..d_head) |r| dot += q[r] * k[s * d_head + r];
        const weight = @exp(dot * scale + mask[s] - max_score);
        denom += weight;
        for (0..d_head) |r| expected[r] += weight * v[s * d_head + r];
    }
    for (&expected) |*value| value.* /= denom;

    const persistent = [_]backend_mod.ProgramIO{
        .host(1, 0, @ptrCast(k.ptr), @intCast(k.len * @sizeOf(f32))),
        .host(2, 0, @ptrCast(v.ptr), @intCast(v.len * @sizeOf(f32))),
        .host(3, 0, @ptrCast(mask.ptr), @intCast(mask.len * @sizeOf(f32))),
    };
    const inputs = [_]backend_mod.ProgramIO{.host(0, 0, @ptrCast(&q), q.len * @sizeOf(f32))};
    const outputs = [_]backend_mod.ProgramIO{.host(4, 0, @ptrCast(&output), output.len * @sizeOf(f32))};

    try std.testing.expect(be.configureBindings(handle, runtime, &persistent, &inputs, &outputs));
    be.uploadBindings(handle, runtime, &persistent);
    be.executeConfiguredBindings(handle, runtime, true);
    for (expected, output) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.005);
}

test "wgpu executable capabilities remain shape-aware" {
    if (!options.use_wgpu) return error.SkipZigTest;

    var backend = WgpuBackend.init(std.testing.allocator);
    defer backend.deinit();
    const be = backend.backend();

    const matmul_ops = [_]backend_mod.DeviceOp{.{ .matmul = .{
        .dst = 2,
        .a = 0,
        .b = 1,
        .geom = .{
            .M = 1,
            .N = 2,
            .K = 2,
            .a_row_stride = 2,
            .a_col_stride = 1,
            .b_row_stride = 2,
            .b_col_stride = 1,
            .a_offset = 0,
            .b_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 2,
        },
    } }};
    const matmul_program = backend_mod.DeviceProgram{
        .ops = &matmul_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 2, 4, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(matmul_program));

    const matvec_elementwise_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .elementwise = .{ .op = .mul, .dst = 4, .src0 = 2, .src1 = 3, .n = 2 } },
    };
    const matvec_elementwise_program = backend_mod.DeviceProgram{
        .ops = &matvec_elementwise_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 2, 4, 2, 2, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(matvec_elementwise_program));

    const matvec_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 3, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
    };
    const matvec_fused_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .fused_elementwise = .{
            .steps = &matvec_fused_steps,
            .n = 2,
            .dst = 4,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const matvec_fused_program = backend_mod.DeviceProgram{
        .ops = &matvec_fused_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 2, 4, 2, 2, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(matvec_fused_program));

    const too_many_matvec_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 3, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 5, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 6, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 7, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 8, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 9, .secondary_offset = 0 },
    };
    const too_many_matvec_fused_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .fused_elementwise = .{
            .steps = &too_many_matvec_fused_steps,
            .n = 2,
            .dst = 10,
            .src = 2,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const too_many_matvec_fused_program = backend_mod.DeviceProgram{
        .ops = &too_many_matvec_fused_ops,
        .n_buffers = 11,
        .buffer_sizes = &.{ 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(too_many_matvec_fused_program));
    try std.testing.expect(be.compileProgram(too_many_matvec_fused_program) == null);

    const alias_matvec_elementwise_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .elementwise = .{ .op = .mul, .dst = 3, .src0 = 2, .src1 = 3, .n = 2 } },
    };
    const alias_matvec_elementwise_program = backend_mod.DeviceProgram{
        .ops = &alias_matvec_elementwise_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 4, 2, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(alias_matvec_elementwise_program));

    const matvec_slice_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .slice_assign = .{
            .dst = 3,
            .src = 2,
            .rows = 1,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 1,
        } },
    };
    const matvec_slice_program = backend_mod.DeviceProgram{
        .ops = &matvec_slice_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 4, 2, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(matvec_slice_program));

    const alias_matvec_slice_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .slice_assign = .{
            .dst = 2,
            .src = 2,
            .rows = 1,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 1,
        } },
    };
    const alias_matvec_slice_program = backend_mod.DeviceProgram{
        .ops = &alias_matvec_slice_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 2, 4, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(alias_matvec_slice_program));

    const matvec_rope_slice_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .rope = .{
            .dst = 4,
            .src = 2,
            .cos_sin = 3,
            .half_d = 1,
            .seq_len = 1,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 2,
            .cs_cs = 4,
        } },
        .{ .slice_assign = .{
            .dst = 5,
            .src = 4,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const matvec_rope_slice_program = backend_mod.DeviceProgram{
        .ops = &matvec_rope_slice_ops,
        .n_buffers = 6,
        .buffer_sizes = &.{ 2, 4, 2, 4, 2, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(matvec_rope_slice_program));

    const alias_matvec_rope_slice_ops = [_]backend_mod.DeviceOp{
        matvec_rope_slice_ops[0],
        matvec_rope_slice_ops[1],
        .{ .slice_assign = .{
            .dst = 4,
            .src = 4,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const alias_matvec_rope_slice_program = backend_mod.DeviceProgram{
        .ops = &alias_matvec_rope_slice_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 2, 4, 2, 4, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(alias_matvec_rope_slice_program));

    const qdata = [_]i8{ 1, 0, 0, 1 };
    const scales = [_]f32{1};
    const qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 2,
        .cols = 2,
        .block_size = 4,
    }};
    const qmatmul_ops = [_]backend_mod.DeviceOp{.{ .qmatmul = .{
        .dst = 1,
        .input = 0,
        .weight_idx = 0,
        .M = 1,
        .N = 2,
        .K = 2,
    } }};
    const qmatmul_program = backend_mod.DeviceProgram{
        .ops = &qmatmul_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qmatmul_program));

    const qlinear_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .elementwise = .{ .op = .add, .dst = 3, .src0 = 1, .src1 = 2, .n = 2 } },
    };
    const qlinear_program = backend_mod.DeviceProgram{
        .ops = &qlinear_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 2, 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qlinear_program));

    const qmatvec_slice_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .slice_assign = .{
            .dst = 2,
            .src = 1,
            .rows = 1,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 1,
        } },
    };
    const qmatvec_slice_program = backend_mod.DeviceProgram{
        .ops = &qmatvec_slice_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 2, 2, 4 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qmatvec_slice_program));

    const qmatvec_rope_slice_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .rope = .{
            .dst = 2,
            .src = 1,
            .cos_sin = 3,
            .half_d = 1,
            .seq_len = 1,
            .src_off = 0,
            .cs_off = 0,
            .dst_off = 0,
            .src_rs = 1,
            .src_cs = 2,
            .cs_cs = 4,
        } },
        .{ .slice_assign = .{
            .dst = 4,
            .src = 2,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const qmatvec_rope_slice_program = backend_mod.DeviceProgram{
        .ops = &qmatvec_rope_slice_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 2, 2, 2, 4, 4 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qmatvec_rope_slice_program));

    const alias_qmatvec_slice_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .slice_assign = .{
            .dst = 1,
            .src = 1,
            .rows = 1,
            .cols = 2,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 1,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 1,
            .patch_stride = 1,
        } },
    };
    const alias_qmatvec_slice_program = backend_mod.DeviceProgram{
        .ops = &alias_qmatvec_slice_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(alias_qmatvec_slice_program));

    const alias_qmatvec_rope_slice_ops = [_]backend_mod.DeviceOp{
        qmatvec_rope_slice_ops[0],
        qmatvec_rope_slice_ops[1],
        .{ .slice_assign = .{
            .dst = 2,
            .src = 2,
            .rows = 2,
            .cols = 1,
            .dst_base_offset = 0,
            .dst_offset = 0,
            .dst_row_stride = 1,
            .dst_col_stride = 2,
            .src_offset = 0,
            .src_row_stride = 1,
            .src_col_stride = 2,
            .patch_stride = 2,
        } },
    };
    const alias_qmatvec_rope_slice_program = backend_mod.DeviceProgram{
        .ops = &alias_qmatvec_rope_slice_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 2, 2, 4 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(alias_qmatvec_rope_slice_program));

    const qmatvec_elementwise_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .elementwise = .{ .op = .mul, .dst = 3, .src0 = 1, .src1 = 2, .n = 2 } },
    };
    const qmatvec_elementwise_program = backend_mod.DeviceProgram{
        .ops = &qmatvec_elementwise_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 2, 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qmatvec_elementwise_program));

    const qmatvec_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .mul, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
    };
    const qmatvec_fused_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .fused_elementwise = .{
            .steps = &qmatvec_fused_steps,
            .n = 2,
            .dst = 3,
            .src = 1,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const qmatvec_fused_program = backend_mod.DeviceProgram{
        .ops = &qmatvec_fused_ops,
        .n_buffers = 4,
        .buffer_sizes = &.{ 2, 2, 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(qmatvec_fused_program));

    const too_many_qmatvec_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 3, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 4, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 5, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 6, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 7, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 8, .secondary_offset = 0 },
    };
    const too_many_qmatvec_fused_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .fused_elementwise = .{
            .steps = &too_many_qmatvec_fused_steps,
            .n = 2,
            .dst = 9,
            .src = 1,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const too_many_qmatvec_fused_program = backend_mod.DeviceProgram{
        .ops = &too_many_qmatvec_fused_ops,
        .n_buffers = 10,
        .buffer_sizes = &.{ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(!be.supportsProgram(too_many_qmatvec_fused_program));
    try std.testing.expect(be.compileProgram(too_many_qmatvec_fused_program) == null);

    const alias_qmatvec_elementwise_ops = [_]backend_mod.DeviceOp{
        qmatmul_ops[0],
        .{ .elementwise = .{ .op = .mul, .dst = 2, .src0 = 1, .src1 = 2, .n = 2 } },
    };
    const alias_qmatvec_elementwise_program = backend_mod.DeviceProgram{
        .ops = &alias_qmatvec_elementwise_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 2, 2, 2 },
        .initial_uploads = &.{},
        .qweights = &qweights,
    };
    try std.testing.expect(be.supportsProgram(alias_qmatvec_elementwise_program));

    const bad_qweights = [_]backend_mod.QuantizedWeightUpload{.{
        .data = &qdata,
        .scales = &scales,
        .rows = 3,
        .cols = 2,
        .block_size = 4,
    }};
    const bad_qmatmul_program = backend_mod.DeviceProgram{
        .ops = &qmatmul_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 2, 2 },
        .initial_uploads = &.{},
        .qweights = &bad_qweights,
    };
    try std.testing.expect(!be.supportsProgram(bad_qmatmul_program));
    try std.testing.expect(be.compileProgram(bad_qmatmul_program) == null);

    const add_ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .add,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const add_program = backend_mod.DeviceProgram{
        .ops = &add_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 4, 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(add_program));

    const fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
        .{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
    };
    const fused_ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &fused_steps,
        .n = 4,
        .dst = 2,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};
    const fused_program = backend_mod.DeviceProgram{
        .ops = &fused_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 4, 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(fused_program));

    const repeat_ops = [_]backend_mod.DeviceOp{.{ .repeat = .{
        .dst = 1,
        .src = 0,
        .n = 8,
        .src_ne = .{ 2, 1, 1, 1 },
        .dst_ne = .{ 2, 4, 1, 1 },
        .src_strides = .{ 1, 2, 2, 2 },
        .dst_strides = .{ 1, 2, 8, 8 },
    } }};
    const repeat_program = backend_mod.DeviceProgram{
        .ops = &repeat_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 2, 8 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(repeat_program));

    const repeat_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 2, .secondary_offset = 0 },
        .{ .op = .recip, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 },
    };
    const repeat_fused_ops = [_]backend_mod.DeviceOp{
        .{ .repeat = .{
            .dst = 2,
            .src = 1,
            .n = 4,
            .src_ne = .{ 1, 1, 1, 1 },
            .dst_ne = .{ 4, 1, 1, 1 },
            .src_strides = .{ 1, 1, 1, 1 },
            .dst_strides = .{ 1, 4, 4, 4 },
        } },
        .{ .fused_elementwise = .{
            .steps = &repeat_fused_steps,
            .n = 4,
            .dst = 4,
            .src = 0,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const repeat_fused_program = backend_mod.DeviceProgram{
        .ops = &repeat_fused_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 4, 1, 4, 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(repeat_fused_program));

    const bad_repeat_fused_steps = [_]backend_mod.FusedEwStep{
        .{ .op = .add, .is_swapped = false, .secondary_buf = 1, .secondary_offset = 0 },
    };
    const bad_repeat_fused_ops = [_]backend_mod.DeviceOp{
        repeat_fused_ops[0],
        .{ .fused_elementwise = .{
            .steps = &bad_repeat_fused_steps,
            .n = 4,
            .dst = 4,
            .src = 0,
            .dst_offset = 0,
            .src_offset = 0,
        } },
    };
    const bad_repeat_fused_program = backend_mod.DeviceProgram{
        .ops = &bad_repeat_fused_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 4, 1, 4, 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(bad_repeat_fused_program));
    try std.testing.expect(be.compileProgram(bad_repeat_fused_program) == null);

    const too_many_fused_steps = [_]backend_mod.FusedEwStep{.{ .op = .relu, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }} ** 9;
    const too_many_fused_ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &too_many_fused_steps,
        .n = 4,
        .dst = 1,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};
    const too_many_fused_program = backend_mod.DeviceProgram{
        .ops = &too_many_fused_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(too_many_fused_program));
    try std.testing.expect(be.compileProgram(too_many_fused_program) == null);

    const sgn_ops = [_]backend_mod.DeviceOp{.{ .elementwise = .{
        .op = .sgn,
        .dst = 2,
        .src0 = 0,
        .src1 = 1,
        .n = 4,
    } }};
    const sgn_program = backend_mod.DeviceProgram{
        .ops = &sgn_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 4, 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(sgn_program));

    const sgn_fused_steps = [_]backend_mod.FusedEwStep{.{ .op = .sgn, .is_swapped = false, .secondary_buf = 0, .secondary_offset = 0 }};
    const sgn_fused_ops = [_]backend_mod.DeviceOp{.{ .fused_elementwise = .{
        .steps = &sgn_fused_steps,
        .n = 4,
        .dst = 1,
        .src = 0,
        .dst_offset = 0,
        .src_offset = 0,
    } }};
    const sgn_fused_program = backend_mod.DeviceProgram{
        .ops = &sgn_fused_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(sgn_fused_program));

    const linear_ops = [_]backend_mod.DeviceOp{
        matmul_ops[0],
        .{ .elementwise = .{ .op = .add, .dst = 4, .src0 = 2, .src1 = 3, .n = 2 } },
    };
    const linear_program = backend_mod.DeviceProgram{
        .ops = &linear_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 2, 4, 2, 2, 2 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(linear_program));

    const layernorm_ops = [_]backend_mod.DeviceOp{.{ .layernorm = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 4,
    } }};
    const layernorm_program = backend_mod.DeviceProgram{
        .ops = &layernorm_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(layernorm_program));

    const rmsnorm_ops = [_]backend_mod.DeviceOp{.{ .rmsnorm = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 4,
    } }};
    const rmsnorm_program = backend_mod.DeviceProgram{
        .ops = &rmsnorm_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(rmsnorm_program));

    const softmax_ops = [_]backend_mod.DeviceOp{.{ .softmax = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 4,
    } }};
    const softmax_program = backend_mod.DeviceProgram{
        .ops = &softmax_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(softmax_program));

    const logsoftmax_ops = [_]backend_mod.DeviceOp{.{ .logsoftmax = .{
        .dst = 1,
        .src = 0,
        .rows = 1,
        .cols = 4,
    } }};
    const logsoftmax_program = backend_mod.DeviceProgram{
        .ops = &logsoftmax_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(logsoftmax_program));

    const reduce_ops = [_]backend_mod.DeviceOp{.{ .reduce = .{
        .op = .sum,
        .dst = 1,
        .src = 0,
        .n_out = 1,
        .reduce_size = 4,
    } }};
    const reduce_program = backend_mod.DeviceProgram{
        .ops = &reduce_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 1 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(reduce_program));

    const bad_reduce_ops = [_]backend_mod.DeviceOp{.{ .reduce = .{
        .op = .mul,
        .dst = 1,
        .src = 0,
        .n_out = 1,
        .reduce_size = 4,
    } }};
    const bad_reduce_program = backend_mod.DeviceProgram{
        .ops = &bad_reduce_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 1 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(bad_reduce_program));
    try std.testing.expect(be.compileProgram(bad_reduce_program) == null);

    const rope_ops = [_]backend_mod.DeviceOp{.{ .rope = .{
        .dst = 2,
        .src = 0,
        .cos_sin = 1,
        .half_d = 2,
        .seq_len = 1,
        .src_off = 0,
        .cs_off = 0,
        .dst_off = 0,
        .src_rs = 1,
        .src_cs = 4,
        .cs_cs = 8,
    } }};
    const rope_program = backend_mod.DeviceProgram{
        .ops = &rope_ops,
        .n_buffers = 3,
        .buffer_sizes = &.{ 4, 8, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(rope_program));

    const slice_ops = [_]backend_mod.DeviceOp{.{ .slice_assign = .{
        .dst = 1,
        .src = 0,
        .rows = 2,
        .cols = 2,
        .dst_base_offset = 0,
        .dst_offset = 0,
        .dst_row_stride = 1,
        .dst_col_stride = 2,
        .src_offset = 0,
        .src_row_stride = 1,
        .src_col_stride = 2,
        .patch_stride = 2,
    } }};
    const slice_program = backend_mod.DeviceProgram{
        .ops = &slice_ops,
        .n_buffers = 2,
        .buffer_sizes = &.{ 4, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(slice_program));

    const attention_ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = 2,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = 2,
        .dst_rs = 1,
        .dst_cs = 4,
    } }};
    const attention_program = backend_mod.DeviceProgram{
        .ops = &attention_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 4, 8, 8, 2, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(attention_program));

    const long_attention_seq_kv = 512;
    const long_attention_ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = long_attention_seq_kv,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = long_attention_seq_kv,
        .dst_rs = 1,
        .dst_cs = 4,
    } }};
    const long_attention_program = backend_mod.DeviceProgram{
        .ops = &long_attention_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 4, long_attention_seq_kv * 4, long_attention_seq_kv * 4, long_attention_seq_kv, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(be.supportsProgram(long_attention_program));

    const too_long_attention_ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 4,
        .seq_q = 1,
        .seq_kv = max_attention_seq_kv + 1,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 4,
        .k_rs = 1,
        .k_cs = 4,
        .v_rs = 1,
        .v_cs = 4,
        .mask_rs = 1,
        .mask_cs = max_attention_seq_kv + 1,
        .dst_rs = 1,
        .dst_cs = 4,
    } }};
    const too_long_attention_program = backend_mod.DeviceProgram{
        .ops = &too_long_attention_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 4, (max_attention_seq_kv + 1) * 4, (max_attention_seq_kv + 1) * 4, max_attention_seq_kv + 1, 4 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(too_long_attention_program));
    try std.testing.expect(be.compileProgram(too_long_attention_program) == null);

    const too_wide_attention_ops = [_]backend_mod.DeviceOp{.{ .attention = .{
        .dst = 4,
        .q = 0,
        .k = 1,
        .v = 2,
        .mask = 3,
        .has_mask = true,
        .d_head = 257,
        .seq_q = 1,
        .seq_kv = 2,
        .scale = 0.5,
        .q_off = 0,
        .k_off = 0,
        .v_off = 0,
        .mask_off = 0,
        .dst_off = 0,
        .q_rs = 1,
        .q_cs = 257,
        .k_rs = 1,
        .k_cs = 257,
        .v_rs = 1,
        .v_cs = 257,
        .mask_rs = 1,
        .mask_cs = 2,
        .dst_rs = 1,
        .dst_cs = 257,
    } }};
    const too_wide_attention_program = backend_mod.DeviceProgram{
        .ops = &too_wide_attention_ops,
        .n_buffers = 5,
        .buffer_sizes = &.{ 257, 514, 514, 2, 257 },
        .initial_uploads = &.{},
    };
    try std.testing.expect(!be.supportsProgram(too_wide_attention_program));
    try std.testing.expect(be.compileProgram(too_wide_attention_program) == null);
}
