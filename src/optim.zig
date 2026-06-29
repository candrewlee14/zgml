//! Native imperative optimizer helpers over trainable Tensor parameters.

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
const max_dims = tensor_mod.max_dims;
const Alloc = std.mem.Allocator;
const OptimizerInitError = Alloc.Error || error{InvalidOptimizerConfig};

pub const native_helper_manifest = .{
    .kind = "zgml-native-helper",
    .domain = "optim",
    .product_frontend = false,
    .product_owner = "src/ts/**",
    .product_policy = "required-core",
    .role = "native-helper-substrate",
    .alignment_boundary = "JS/TS API -> Zig C ABI -> Program/Session kernels",
};

pub const OptimizerKind = enum {
    sgd,
    adam,
    adamw,
};

pub const StateLoadError = error{
    OptimizerKindMismatch,
    ParameterCountMismatch,
    MissingState,
    UnexpectedState,
    ShapeMismatch,
    LayoutMismatch,
};

pub fn StateEntry(comptime T: type) type {
    return struct {
        name: []const u8,
        n_dims: u8,
        ne: [max_dims]usize,
        layout: ?[]const u8 = null,
        data: []T,

        pub fn shape(self: *const @This()) []const usize {
            return self.ne[0..@as(usize, self.n_dims)];
        }
    };
}

pub fn StateDict(comptime T: type) type {
    return struct {
        kind: OptimizerKind,
        step: usize,
        param_count: usize,
        entries: []StateEntry(T),

        pub fn deinit(self: *@This(), alloc: Alloc) void {
            deinitStateDict(T, alloc, self);
        }
    };
}

pub const SGDConfig = struct {
    lr: f64 = 0.01,
    momentum: f64 = 0.0,
    weight_decay: f64 = 0.0,
};

pub const AdamConfig = struct {
    lr: f64 = 1e-3,
    beta1: f64 = 0.9,
    beta2: f64 = 0.999,
    eps: f64 = 1e-8,
    weight_decay: f64 = 0.0,
};

pub const AdamWConfig = struct {
    lr: f64 = 1e-3,
    beta1: f64 = 0.9,
    beta2: f64 = 0.999,
    eps: f64 = 1e-8,
    weight_decay: f64 = 0.0,
};

pub fn zeroGrad(comptime T: type, params: []const *Tensor(T)) void {
    for (params) |param| {
        if (param.grad) |grad| _ = grad.setAllScalar(0);
    }
}

pub fn stateDict(comptime T: type, alloc: Alloc, optimizer: anytype) Alloc.Error!StateDict(T) {
    return optimizer.stateDict(alloc);
}

pub fn deinitStateDict(comptime T: type, alloc: Alloc, state: *StateDict(T)) void {
    freeStateEntryPayloads(T, alloc, state.entries);
    alloc.free(state.entries);
    state.entries = &.{};
}

pub fn loadStateDict(comptime T: type, optimizer: anytype, state: StateDict(T)) StateLoadError!void {
    try optimizer.loadStateDict(state);
}

pub fn SGD(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        params: []*Tensor(T),
        lr: T,
        momentum: T,
        weight_decay: T,
        velocity: []*Tensor(T),

        pub fn init(alloc: Alloc, params: []const *Tensor(T), config: SGDConfig) OptimizerInitError!Self {
            try validateSGDConfig(config);
            const momentum: T = @floatCast(config.momentum);
            const owned_params = try alloc.dupe(*Tensor(T), params);
            errdefer alloc.free(owned_params);
            const velocity = try allocState(T, alloc, owned_params, momentum != 0);
            return .{
                .alloc = alloc,
                .params = owned_params,
                .lr = @floatCast(config.lr),
                .momentum = momentum,
                .weight_decay = @floatCast(config.weight_decay),
                .velocity = velocity,
            };
        }

        pub fn deinit(self: *Self) void {
            freeState(T, self.alloc, self.velocity);
            self.alloc.free(self.params);
        }

        pub fn step(self: *Self) void {
            if (self.momentum == 0) {
                for (self.params) |param| {
                    const grad = param.grad orelse continue;
                    for (param.data, grad.data) |*p, g| {
                        p.* -= self.lr * (g + self.weight_decay * p.*);
                    }
                }
                return;
            }

            for (self.params, self.velocity) |param, velocity| {
                const grad = param.grad orelse continue;
                for (param.data, grad.data, velocity.data) |*p, g, *v| {
                    v.* = self.momentum * v.* + g + self.weight_decay * p.*;
                    p.* -= self.lr * v.*;
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            optim.zeroGrad(T, self.params);
        }

        pub fn stateDict(self: *Self, alloc: Alloc) Alloc.Error!StateDict(T) {
            const entries = try allocStateDictEntries(T, alloc, self.velocity.len);
            var initialized: usize = 0;
            errdefer {
                freeStateEntryPayloads(T, alloc, entries[0..initialized]);
                alloc.free(entries);
            }

            for (self.params[0..self.velocity.len], self.velocity, 0..) |param, velocity, index| {
                entries[initialized] = try makeStateEntry(T, alloc, "velocity", index, param, velocity);
                initialized += 1;
            }

            return .{
                .kind = .sgd,
                .step = 0,
                .param_count = self.params.len,
                .entries = entries,
            };
        }

        pub fn loadStateDict(self: *Self, state: StateDict(T)) StateLoadError!void {
            try validateStateHeader(T, state, .sgd, self.params.len);
            if (self.velocity.len == 0) {
                if (state.entries.len != 0) return error.UnexpectedState;
                return;
            }

            try validateIndexedStateGroup(T, state.entries, "velocity", self.params);
            try rejectUnexpectedIndexedState(T, state.entries, &.{"velocity"}, self.params.len);
            copyIndexedStateGroup(T, state.entries, "velocity", self.velocity);
        }
    };
}

pub fn Adam(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        params: []*Tensor(T),
        config: AdamConfig,
        t: usize,
        m: []*Tensor(T),
        v: []*Tensor(T),

        pub fn init(alloc: Alloc, params: []const *Tensor(T), config: AdamConfig) OptimizerInitError!Self {
            try validateAdamConfig(config);
            const owned_params = try alloc.dupe(*Tensor(T), params);
            errdefer alloc.free(owned_params);
            const m = try allocState(T, alloc, owned_params, true);
            errdefer freeState(T, alloc, m);
            const v = try allocState(T, alloc, owned_params, true);
            return .{
                .alloc = alloc,
                .params = owned_params,
                .config = config,
                .t = 0,
                .m = m,
                .v = v,
            };
        }

        pub fn deinit(self: *Self) void {
            freeState(T, self.alloc, self.v);
            freeState(T, self.alloc, self.m);
            self.alloc.free(self.params);
        }

        pub fn step(self: *Self) void {
            self.t += 1;

            const beta1: T = @floatCast(self.config.beta1);
            const beta2: T = @floatCast(self.config.beta2);
            const lr: T = @floatCast(self.config.lr);
            const eps: T = @floatCast(self.config.eps);
            const wd: T = @floatCast(self.config.weight_decay);
            const t_float: T = @floatFromInt(self.t);
            const bias_correction1 = 1 / (1 - std.math.pow(T, beta1, t_float));
            const bias_correction2 = 1 / (1 - std.math.pow(T, beta2, t_float));

            for (self.params, self.m, self.v) |param, m_t, v_t| {
                const grad = param.grad orelse continue;
                for (param.data, grad.data, m_t.data, v_t.data) |*p, g, *m, *v| {
                    const g_with_decay = g + wd * p.*;
                    m.* = beta1 * m.* + (1 - beta1) * g_with_decay;
                    v.* = beta2 * v.* + (1 - beta2) * g_with_decay * g_with_decay;

                    const m_hat = m.* * bias_correction1;
                    const v_hat = v.* * bias_correction2;
                    p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            optim.zeroGrad(T, self.params);
        }

        pub fn stateDict(self: *Self, alloc: Alloc) Alloc.Error!StateDict(T) {
            return adamLikeStateDict(T, alloc, .adam, self.t, self.params, self.m, self.v);
        }

        pub fn loadStateDict(self: *Self, state: StateDict(T)) StateLoadError!void {
            try validateStateHeader(T, state, .adam, self.params.len);
            try validateIndexedStateGroup(T, state.entries, "m", self.params);
            try validateIndexedStateGroup(T, state.entries, "v", self.params);
            try rejectUnexpectedIndexedState(T, state.entries, &.{ "m", "v" }, self.params.len);
            copyIndexedStateGroup(T, state.entries, "m", self.m);
            copyIndexedStateGroup(T, state.entries, "v", self.v);
            self.t = state.step;
        }
    };
}

pub fn AdamW(comptime T: type) type {
    return struct {
        const Self = @This();

        alloc: Alloc,
        params: []*Tensor(T),
        config: AdamWConfig,
        t: usize,
        m: []*Tensor(T),
        v: []*Tensor(T),

        pub fn init(alloc: Alloc, params: []const *Tensor(T), config: AdamWConfig) OptimizerInitError!Self {
            try validateAdamWConfig(config);
            const owned_params = try alloc.dupe(*Tensor(T), params);
            errdefer alloc.free(owned_params);
            const m = try allocState(T, alloc, owned_params, true);
            errdefer freeState(T, alloc, m);
            const v = try allocState(T, alloc, owned_params, true);
            return .{
                .alloc = alloc,
                .params = owned_params,
                .config = config,
                .t = 0,
                .m = m,
                .v = v,
            };
        }

        pub fn deinit(self: *Self) void {
            freeState(T, self.alloc, self.v);
            freeState(T, self.alloc, self.m);
            self.alloc.free(self.params);
        }

        pub fn step(self: *Self) void {
            self.t += 1;

            const beta1: T = @floatCast(self.config.beta1);
            const beta2: T = @floatCast(self.config.beta2);
            const lr: T = @floatCast(self.config.lr);
            const eps: T = @floatCast(self.config.eps);
            const wd: T = @floatCast(self.config.weight_decay);
            const t_float: T = @floatFromInt(self.t);
            const bias_correction1 = 1 / (1 - std.math.pow(T, beta1, t_float));
            const bias_correction2 = 1 / (1 - std.math.pow(T, beta2, t_float));

            for (self.params, self.m, self.v) |param, m_t, v_t| {
                const grad = param.grad orelse continue;
                for (param.data, grad.data, m_t.data, v_t.data) |*p, g, *m, *v| {
                    m.* = beta1 * m.* + (1 - beta1) * g;
                    v.* = beta2 * v.* + (1 - beta2) * g * g;

                    const m_hat = m.* * bias_correction1;
                    const v_hat = v.* * bias_correction2;
                    if (wd != 0) p.* -= lr * wd * p.*;
                    p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
                }
            }
        }

        pub fn zeroGrad(self: *Self) void {
            optim.zeroGrad(T, self.params);
        }

        pub fn stateDict(self: *Self, alloc: Alloc) Alloc.Error!StateDict(T) {
            return adamLikeStateDict(T, alloc, .adamw, self.t, self.params, self.m, self.v);
        }

        pub fn loadStateDict(self: *Self, state: StateDict(T)) StateLoadError!void {
            try validateStateHeader(T, state, .adamw, self.params.len);
            try validateIndexedStateGroup(T, state.entries, "m", self.params);
            try validateIndexedStateGroup(T, state.entries, "v", self.params);
            try rejectUnexpectedIndexedState(T, state.entries, &.{ "m", "v" }, self.params.len);
            copyIndexedStateGroup(T, state.entries, "m", self.m);
            copyIndexedStateGroup(T, state.entries, "v", self.v);
            self.t = state.step;
        }
    };
}

const optim = @This();

fn validNonNegative(x: f64) bool {
    return std.math.isFinite(x) and x >= 0;
}

fn validLessThanOne(x: f64) bool {
    return validNonNegative(x) and x < 1;
}

fn validateSGDConfig(config: SGDConfig) error{InvalidOptimizerConfig}!void {
    if (!validNonNegative(config.lr)) return error.InvalidOptimizerConfig;
    if (!validLessThanOne(config.momentum)) return error.InvalidOptimizerConfig;
    if (!validNonNegative(config.weight_decay)) return error.InvalidOptimizerConfig;
}

fn validateAdamConfig(config: AdamConfig) error{InvalidOptimizerConfig}!void {
    if (!validNonNegative(config.lr)) return error.InvalidOptimizerConfig;
    if (!validLessThanOne(config.beta1)) return error.InvalidOptimizerConfig;
    if (!validLessThanOne(config.beta2)) return error.InvalidOptimizerConfig;
    if (!std.math.isFinite(config.eps) or config.eps <= 0) return error.InvalidOptimizerConfig;
    if (!validNonNegative(config.weight_decay)) return error.InvalidOptimizerConfig;
}

fn validateAdamWConfig(config: AdamWConfig) error{InvalidOptimizerConfig}!void {
    if (!validNonNegative(config.lr)) return error.InvalidOptimizerConfig;
    if (!validLessThanOne(config.beta1)) return error.InvalidOptimizerConfig;
    if (!validLessThanOne(config.beta2)) return error.InvalidOptimizerConfig;
    if (!std.math.isFinite(config.eps) or config.eps <= 0) return error.InvalidOptimizerConfig;
    if (!validNonNegative(config.weight_decay)) return error.InvalidOptimizerConfig;
}

fn allocState(comptime T: type, alloc: Alloc, params: []const *Tensor(T), active: bool) Alloc.Error![]*Tensor(T) {
    const len = if (active) params.len else 0;
    const state = try alloc.alloc(*Tensor(T), len);
    var initialized: usize = 0;
    errdefer {
        for (state[0..initialized]) |tensor| tensor.deinit();
        alloc.free(state);
    }

    for (params[0..len]) |param| {
        const tensor = try Tensor(T).init(alloc, param.ne[0..param.n_dims]);
        _ = tensor.setAllScalar(0);
        state[initialized] = tensor;
        initialized += 1;
    }
    return state;
}

fn freeState(comptime T: type, alloc: Alloc, state: []*Tensor(T)) void {
    for (state) |tensor| tensor.deinit();
    alloc.free(state);
}

fn allocStateDictEntries(comptime T: type, alloc: Alloc, len: usize) Alloc.Error![]StateEntry(T) {
    return alloc.alloc(StateEntry(T), len);
}

fn makeStateEntry(comptime T: type, alloc: Alloc, comptime prefix: []const u8, index: usize, param: *const Tensor(T), source: *const Tensor(T)) Alloc.Error!StateEntry(T) {
    const name = try std.fmt.allocPrint(alloc, "{s}.{d}", .{ prefix, index });
    errdefer alloc.free(name);
    const layout = try alloc.dupe(u8, "dense");
    errdefer alloc.free(layout);
    const data = try alloc.dupe(T, source.denseSliceConst());
    errdefer alloc.free(data);
    return .{
        .name = name,
        .n_dims = param.n_dims,
        .ne = param.ne,
        .layout = layout,
        .data = data,
    };
}

fn freeStateEntryPayloads(comptime T: type, alloc: Alloc, entries: []StateEntry(T)) void {
    for (entries) |entry| {
        alloc.free(entry.name);
        if (entry.layout) |layout| alloc.free(layout);
        alloc.free(entry.data);
    }
}

fn adamLikeStateDict(
    comptime T: type,
    alloc: Alloc,
    kind: OptimizerKind,
    step: usize,
    params: []const *Tensor(T),
    m: []const *Tensor(T),
    v: []const *Tensor(T),
) Alloc.Error!StateDict(T) {
    const entries = try allocStateDictEntries(T, alloc, params.len * 2);
    var initialized: usize = 0;
    errdefer {
        freeStateEntryPayloads(T, alloc, entries[0..initialized]);
        alloc.free(entries);
    }

    for (params, m, v, 0..) |param, m_t, v_t, index| {
        entries[initialized] = try makeStateEntry(T, alloc, "m", index, param, m_t);
        initialized += 1;
        entries[initialized] = try makeStateEntry(T, alloc, "v", index, param, v_t);
        initialized += 1;
    }

    return .{
        .kind = kind,
        .step = step,
        .param_count = params.len,
        .entries = entries,
    };
}

fn validateStateHeader(comptime T: type, state: StateDict(T), expected_kind: OptimizerKind, expected_param_count: usize) StateLoadError!void {
    if (state.kind != expected_kind) return error.OptimizerKindMismatch;
    if (state.param_count != expected_param_count) return error.ParameterCountMismatch;
}

fn validateIndexedStateGroup(
    comptime T: type,
    entries: []const StateEntry(T),
    comptime prefix: []const u8,
    params: []const *Tensor(T),
) StateLoadError!void {
    for (params, 0..) |param, index| {
        const entry = findIndexedStateEntry(T, entries, prefix, index) orelse return error.MissingState;
        if (!stateEntryMatchesTensor(T, entry, param)) return error.ShapeMismatch;
        if (!stateEntryMatchesLayout(T, entry, param)) return error.LayoutMismatch;
    }
}

fn copyIndexedStateGroup(
    comptime T: type,
    entries: []const StateEntry(T),
    comptime prefix: []const u8,
    targets: []const *Tensor(T),
) void {
    for (targets, 0..) |target, index| {
        const entry = findIndexedStateEntry(T, entries, prefix, index).?;
        target.setData(entry.data);
    }
}

fn rejectUnexpectedIndexedState(comptime T: type, entries: []const StateEntry(T), comptime prefixes: []const []const u8, param_count: usize) StateLoadError!void {
    for (entries) |entry| {
        if (!isExpectedIndexedStateName(entry.name, prefixes, param_count)) return error.UnexpectedState;
    }

    for (entries, 0..) |entry, index| {
        for (entries[index + 1 ..]) |other| {
            if (std.mem.eql(u8, entry.name, other.name)) return error.UnexpectedState;
        }
    }
}

fn isExpectedIndexedStateName(name: []const u8, comptime prefixes: []const []const u8, param_count: usize) bool {
    inline for (prefixes) |prefix| {
        if (parseIndexedStateName(name, prefix)) |index| {
            if (index < param_count) return true;
        }
    }
    return false;
}

fn findIndexedStateEntry(comptime T: type, entries: []const StateEntry(T), comptime prefix: []const u8, index: usize) ?StateEntry(T) {
    for (entries) |entry| {
        if (parseIndexedStateName(entry.name, prefix)) |entry_index| {
            if (entry_index == index) return entry;
        }
    }
    return null;
}

fn parseIndexedStateName(name: []const u8, comptime prefix: []const u8) ?usize {
    const marker_len = prefix.len + 1;
    if (name.len <= marker_len) return null;
    if (!std.mem.eql(u8, name[0..prefix.len], prefix)) return null;
    if (name[prefix.len] != '.') return null;
    return std.fmt.parseUnsigned(usize, name[marker_len..], 10) catch null;
}

fn stateEntryMatchesTensor(comptime T: type, entry: StateEntry(T), tensor: *const Tensor(T)) bool {
    if (entry.n_dims != tensor.n_dims) return false;
    if (entry.data.len != tensor.nElems()) return false;
    return tensor.hasShape(entry.shape());
}

fn stateEntryMatchesLayout(comptime T: type, entry: StateEntry(T), tensor: *const Tensor(T)) bool {
    _ = tensor;
    const layout = entry.layout orelse return true;
    return std.mem.eql(u8, layout, "dense");
}

const testing = std.testing;
const tac = testing.allocator;
const ComputeGraph = @import("graph.zig").ComputeGraph;

test "optim native helper surface stays curated" {
    const root = @This();
    const expected = .{
        "native_helper_manifest",
        "SGDConfig",
        "AdamConfig",
        "AdamWConfig",
        "OptimizerKind",
        "StateLoadError",
        "StateEntry",
        "StateDict",
        "zeroGrad",
        "stateDict",
        "deinitStateDict",
        "loadStateDict",
        "SGD",
        "Adam",
        "AdamW",
    };
    try testing.expectEqual(expected.len, @typeInfo(root).@"struct".decls.len);
    inline for (expected) |name| {
        try testing.expect(@hasDecl(root, name));
    }
    try testing.expectEqualStrings("zgml-native-helper", native_helper_manifest.kind);
    try testing.expectEqualStrings("optim", native_helper_manifest.domain);
    try testing.expectEqual(false, native_helper_manifest.product_frontend);
    try testing.expectEqualStrings("src/ts/**", native_helper_manifest.product_owner);
    try testing.expectEqualStrings("required-core", native_helper_manifest.product_policy);
    try testing.expectEqualStrings("native-helper-substrate", native_helper_manifest.role);
    try testing.expectEqualStrings("JS/TS API -> Zig C ABI -> Program/Session kernels", native_helper_manifest.alignment_boundary);
}

test "zeroGrad clears parameter gradients" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setParam();
    x.grad.?.setData(&.{ 3, -4 });

    zeroGrad(f32, &.{x});

    try testing.expectEqualSlices(f32, &.{ 0, 0 }, x.grad.?.data);
}

test "sgd converges on scalar quadratic" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{1});
    x.setData(&.{3});
    x.setParam();
    const loss = x.sqr().sumAll();

    var opt = try SGD(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer opt.deinit();

    for (0..60) |_| {
        try g.run(loss);
        opt.step();
    }

    try testing.expect(@abs(x.data[0]) < 1e-4);
}

test "sgd momentum allocates state only when requested" {
    const x = try Tensor(f32).init(tac, &.{1});
    defer x.deinit();
    x.setData(&.{1});
    x.setParam();
    x.grad.?.setData(&.{2});

    var plain = try SGD(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer plain.deinit();
    try testing.expectEqual(@as(usize, 0), plain.velocity.len);

    var momentum = try SGD(f32).init(tac, &.{x}, .{ .lr = 0.1, .momentum = 0.9 });
    defer momentum.deinit();
    try testing.expectEqual(@as(usize, 1), momentum.velocity.len);
}

test "sgd stateDict snapshots and loadStateDict restores momentum" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setData(&.{ 1, -1 });
    x.setParam();
    x.grad.?.setData(&.{ 2, -4 });

    var opt = try SGD(f32).init(tac, &.{x}, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt.deinit();
    opt.step();

    var state = try opt.stateDict(tac);
    defer state.deinit(tac);

    try testing.expectEqual(OptimizerKind.sgd, state.kind);
    try testing.expectEqual(@as(usize, 0), state.step);
    try testing.expectEqual(@as(usize, 1), state.param_count);
    try testing.expectEqual(@as(usize, 1), state.entries.len);
    try testing.expectEqualStrings("velocity.0", state.entries[0].name);
    try testing.expectEqualStrings("dense", state.entries[0].layout.?);
    try testing.expectEqualSlices(usize, &.{2}, state.entries[0].shape());
    try testing.expectEqualSlices(f32, &.{ 2, -4 }, state.entries[0].data);

    var restored = try SGD(f32).init(tac, &.{x}, .{ .lr = 0.1, .momentum = 0.9 });
    defer restored.deinit();
    try loadStateDict(f32, &restored, state);
    try testing.expectEqualSlices(f32, state.entries[0].data, restored.velocity[0].denseSliceConst());
}

test "sgd rejects invalid configs" {
    const x = try Tensor(f32).init(tac, &.{1});
    defer x.deinit();
    x.setParam();

    try testing.expectError(error.InvalidOptimizerConfig, SGD(f32).init(tac, &.{x}, .{ .lr = -0.1 }));
    try testing.expectError(error.InvalidOptimizerConfig, SGD(f32).init(tac, &.{x}, .{ .momentum = 1.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, SGD(f32).init(tac, &.{x}, .{ .weight_decay = -0.1 }));
}

test "sgd owns parameter slice and skips missing gradients" {
    const x = try Tensor(f32).init(tac, &.{1});
    defer x.deinit();
    x.setData(&.{1});
    x.setParam();
    x.grad.?.setData(&.{2});

    const frozen = try Tensor(f32).init(tac, &.{1});
    defer frozen.deinit();
    frozen.setData(&.{7});

    var params = [_]*Tensor(f32){ x, frozen };
    var opt = try SGD(f32).init(tac, params[0..1], .{ .lr = 0.1 });
    defer opt.deinit();
    params[0] = frozen;

    opt.step();

    try testing.expectApproxEqAbs(@as(f32, 0.8), x.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7), frozen.data[0], 1e-6);
}

test "adam converges on scalar quadratic" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{1});
    x.setData(&.{3});
    x.setParam();
    const loss = x.sqr().sumAll();

    var opt = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer opt.deinit();

    for (0..100) |_| {
        try g.run(loss);
        opt.step();
    }

    try testing.expect(@abs(x.data[0]) < 0.1);
}

test "adam uses coupled weight decay" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setData(&.{ 2, -2 });
    x.setParam();
    _ = x.grad.?.setAllScalar(0);

    var opt = try Adam(f32).init(tac, &.{x}, .{
        .lr = 0.1,
        .beta1 = 0,
        .beta2 = 0,
        .eps = 1e-8,
        .weight_decay = 0.1,
    });
    defer opt.deinit();
    opt.step();

    try testing.expectApproxEqAbs(@as(f32, 1.9), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.9), x.data[1], 1e-5);
}

test "adam rejects invalid configs" {
    const x = try Tensor(f32).init(tac, &.{1});
    defer x.deinit();
    x.setParam();

    try testing.expectError(error.InvalidOptimizerConfig, Adam(f32).init(tac, &.{x}, .{ .lr = -0.1 }));
    try testing.expectError(error.InvalidOptimizerConfig, Adam(f32).init(tac, &.{x}, .{ .beta1 = 1.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, Adam(f32).init(tac, &.{x}, .{ .beta2 = 1.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, Adam(f32).init(tac, &.{x}, .{ .eps = 0.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, Adam(f32).init(tac, &.{x}, .{ .weight_decay = -0.1 }));
}

test "adam loadStateDict rejects incompatible snapshots" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setParam();
    x.grad.?.setData(&.{ 1, 2 });

    var opt = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer opt.deinit();
    opt.step();

    var state = try opt.stateDict(tac);
    defer state.deinit(tac);

    var wrong_kind = state;
    wrong_kind.kind = .adamw;
    try testing.expectError(error.OptimizerKindMismatch, opt.loadStateDict(wrong_kind));

    var wrong_count = state;
    wrong_count.param_count = 2;
    try testing.expectError(error.ParameterCountMismatch, opt.loadStateDict(wrong_count));

    var missing_entries: [0]StateEntry(f32) = .{};
    const missing_state = StateDict(f32){
        .kind = .adam,
        .step = state.step,
        .param_count = state.param_count,
        .entries = missing_entries[0..],
    };
    try testing.expectError(error.MissingState, opt.loadStateDict(missing_state));

    var wrong_shape = state;
    const original_n_dims = state.entries[0].n_dims;
    const original_ne = state.entries[0].ne;
    wrong_shape.entries[0].n_dims = 1;
    wrong_shape.entries[0].ne[0] = 1;
    try testing.expectError(error.ShapeMismatch, opt.loadStateDict(wrong_shape));
    wrong_shape.entries[0].n_dims = original_n_dims;
    wrong_shape.entries[0].ne = original_ne;

    var wrong_layout_entries = [_]StateEntry(f32){ state.entries[0], state.entries[1] };
    wrong_layout_entries[0].layout = "strided";
    const wrong_layout = StateDict(f32){
        .kind = .adam,
        .step = state.step,
        .param_count = state.param_count,
        .entries = wrong_layout_entries[0..],
    };
    try testing.expectError(error.LayoutMismatch, opt.loadStateDict(wrong_layout));

    var duplicate_entries: [3]StateEntry(f32) = .{ state.entries[0], state.entries[1], state.entries[0] };
    const duplicate_state = StateDict(f32){
        .kind = .adam,
        .step = state.step,
        .param_count = state.param_count,
        .entries = duplicate_entries[0..],
    };
    try testing.expectError(error.UnexpectedState, opt.loadStateDict(duplicate_state));

    var extra_entries: [3]StateEntry(f32) = .{ state.entries[0], state.entries[1], state.entries[0] };
    extra_entries[2].name = "extra.0";
    const extra_state = StateDict(f32){
        .kind = .adam,
        .step = state.step,
        .param_count = state.param_count,
        .entries = extra_entries[0..],
    };
    try testing.expectError(error.UnexpectedState, opt.loadStateDict(extra_state));
}

test "adam loadStateDict does not partially mutate moments on error" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setParam();
    x.grad.?.setData(&.{ 1, 2 });

    var source = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer source.deinit();
    source.step();

    var state = try source.stateDict(tac);
    defer state.deinit(tac);

    var target = try Adam(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer target.deinit();
    target.t = 42;
    target.m[0].setData(&.{ 5, 6 });
    target.v[0].setData(&.{ 7, 8 });

    var bad_entries = [_]StateEntry(f32){ state.entries[0], state.entries[1] };
    bad_entries[1].n_dims = 1;
    bad_entries[1].ne[0] = 1;
    const bad_state = StateDict(f32){
        .kind = .adam,
        .step = 99,
        .param_count = state.param_count,
        .entries = bad_entries[0..],
    };

    try testing.expectError(error.ShapeMismatch, target.loadStateDict(bad_state));
    try testing.expectEqual(@as(usize, 42), target.t);
    try testing.expectEqualSlices(f32, &.{ 5, 6 }, target.m[0].denseSliceConst());
    try testing.expectEqualSlices(f32, &.{ 7, 8 }, target.v[0].denseSliceConst());
}

test "adamw converges on scalar quadratic" {
    var g = ComputeGraph(f32).init(tac);
    defer g.deinit();
    const a = g.allocator();

    const x = try Tensor(f32).init(a, &.{1});
    x.setData(&.{3});
    x.setParam();
    const loss = x.sqr().sumAll();

    var opt = try AdamW(f32).init(tac, &.{x}, .{ .lr = 0.1 });
    defer opt.deinit();

    for (0..100) |_| {
        try g.run(loss);
        opt.step();
    }

    try testing.expect(@abs(x.data[0]) < 0.1);
}

test "adamw decoupled weight decay shrinks parameters with zero gradient" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setData(&.{ 10, -10 });
    x.setParam();
    _ = x.grad.?.setAllScalar(0);

    var opt = try AdamW(f32).init(tac, &.{x}, .{ .lr = 0.1, .weight_decay = 0.1 });
    defer opt.deinit();
    opt.step();

    try testing.expectApproxEqAbs(@as(f32, 9.9), x.data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -9.9), x.data[1], 1e-5);
}

test "adamw stateDict snapshots and loadStateDict restores moments and step" {
    const x = try Tensor(f32).init(tac, &.{2});
    defer x.deinit();
    x.setData(&.{ 1, -1 });
    x.setParam();
    x.grad.?.setData(&.{ 2, -4 });

    var opt = try AdamW(f32).init(tac, &.{x}, .{
        .lr = 0.1,
        .beta1 = 0.5,
        .beta2 = 0.25,
    });
    defer opt.deinit();
    opt.step();

    var state = try optim.stateDict(f32, tac, &opt);
    defer optim.deinitStateDict(f32, tac, &state);

    try testing.expectEqual(OptimizerKind.adamw, state.kind);
    try testing.expectEqual(@as(usize, 1), state.step);
    try testing.expectEqual(@as(usize, 1), state.param_count);
    try testing.expectEqual(@as(usize, 2), state.entries.len);
    try testing.expectEqualStrings("m.0", state.entries[0].name);
    try testing.expectEqualStrings("v.0", state.entries[1].name);
    try testing.expectEqualStrings("dense", state.entries[0].layout.?);
    try testing.expectEqualStrings("dense", state.entries[1].layout.?);
    try testing.expectEqualSlices(usize, &.{2}, state.entries[0].shape());

    var restored = try AdamW(f32).init(tac, &.{x}, .{
        .lr = 0.1,
        .beta1 = 0.5,
        .beta2 = 0.25,
    });
    defer restored.deinit();
    try restored.loadStateDict(state);

    try testing.expectEqual(state.step, restored.t);
    try testing.expectEqualSlices(f32, state.entries[0].data, restored.m[0].denseSliceConst());
    try testing.expectEqualSlices(f32, state.entries[1].data, restored.v[0].denseSliceConst());
}

test "adamw rejects invalid configs" {
    const x = try Tensor(f32).init(tac, &.{1});
    defer x.deinit();
    x.setParam();

    try testing.expectError(error.InvalidOptimizerConfig, AdamW(f32).init(tac, &.{x}, .{ .lr = -0.1 }));
    try testing.expectError(error.InvalidOptimizerConfig, AdamW(f32).init(tac, &.{x}, .{ .beta1 = 1.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, AdamW(f32).init(tac, &.{x}, .{ .beta2 = 1.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, AdamW(f32).init(tac, &.{x}, .{ .eps = 0.0 }));
    try testing.expectError(error.InvalidOptimizerConfig, AdamW(f32).init(tac, &.{x}, .{ .weight_decay = -0.1 }));
}

test "adamw skips missing gradients" {
    const frozen = try Tensor(f32).init(tac, &.{1});
    defer frozen.deinit();
    frozen.setData(&.{10});

    var opt = try AdamW(f32).init(tac, &.{frozen}, .{ .lr = 0.1, .weight_decay = 0.1 });
    defer opt.deinit();
    opt.step();

    try testing.expectApproxEqAbs(@as(f32, 10), frozen.data[0], 1e-6);
}
