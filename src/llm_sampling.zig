const std = @import("std");

pub const top_k_max: usize = 256;

pub const Options = struct {
    top_k: usize = 40,
    seed: u32 = 0,
    temperature: f64 = 1.0,
};

pub fn TokenScore(comptime T: type) type {
    return struct {
        token: usize,
        logit: T,
    };
}

fn Candidate(comptime T: type) type {
    return struct {
        token: usize,
        logit: T,
    };
}

pub fn validateOptions(options: Options) !void {
    if (options.top_k == 0 or options.top_k > top_k_max) return error.InvalidArgument;
    if (!std.math.isFinite(options.temperature) or options.temperature <= 0) return error.InvalidArgument;
}

pub fn argmax(comptime T: type, logits: []const T) !TokenScore(T) {
    if (logits.len == 0) return error.InvalidTokenWindow;
    var best_token: usize = 0;
    var best = logits[0];
    for (logits[1..], 1..) |value, i| {
        if (value > best) {
            best = value;
            best_token = i;
        }
    }
    return .{ .token = best_token, .logit = best };
}

fn insertCandidate(
    comptime T: type,
    candidates: *[top_k_max]Candidate(T),
    count: *usize,
    candidate: Candidate(T),
) void {
    var i = count.*;
    while (i > 0 and candidate.logit > candidates[i - 1].logit) {
        candidates[i] = candidates[i - 1];
        i -= 1;
    }
    candidates[i] = candidate;
    count.* += 1;
}

fn randomUnit(seed: u32) f64 {
    var x = if (seed == 0) @as(u32, 0x9e37_79b9) else seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return @as(f64, @floatFromInt(x)) / 4294967296.0;
}

pub fn sampleTopK(comptime T: type, logits: []const T, options: Options) !TokenScore(T) {
    try validateOptions(options);
    if (logits.len == 0) return error.InvalidTokenWindow;

    const effective_k = @min(options.top_k, logits.len);
    var candidates: [top_k_max]Candidate(T) = undefined;
    var count: usize = 0;

    for (logits, 0..) |value, i| {
        if (!std.math.isFinite(value)) return error.InvalidArgument;
        const candidate = Candidate(T){ .token = i, .logit = value };
        if (count < effective_k) {
            insertCandidate(T, &candidates, &count, candidate);
        } else if (value > candidates[count - 1].logit) {
            count -= 1;
            insertCandidate(T, &candidates, &count, candidate);
        }
    }

    if (effective_k == 1) {
        return .{ .token = candidates[0].token, .logit = candidates[0].logit };
    }

    var weights: [top_k_max]f64 = undefined;
    var total: f64 = 0;
    const max_logit: f64 = @floatCast(candidates[0].logit);
    const inv_temperature = 1.0 / options.temperature;
    for (candidates[0..effective_k], 0..) |candidate, i| {
        const shifted = (@as(f64, @floatCast(candidate.logit)) - max_logit) * inv_temperature;
        const weight = @exp(shifted);
        weights[i] = weight;
        total += weight;
    }

    var threshold = randomUnit(options.seed) * total;
    for (candidates[0..effective_k], 0..) |candidate, i| {
        if (threshold < weights[i]) {
            return .{ .token = candidate.token, .logit = candidate.logit };
        }
        threshold -= weights[i];
    }

    const fallback = candidates[effective_k - 1];
    return .{ .token = fallback.token, .logit = fallback.logit };
}

const testing = std.testing;

test "llm sampling argmax selects highest logit" {
    const logits = [_]f32{ -1.0, 3.0, 2.0 };
    const selected = try argmax(f32, &logits);
    try testing.expectEqual(@as(usize, 1), selected.token);
    try testing.expectEqual(@as(f32, 3.0), selected.logit);
}

test "llm sampling top-k one is deterministic argmax" {
    const logits = [_]f32{ 0.25, 1.5, 0.75 };
    const selected = try sampleTopK(f32, &logits, .{ .top_k = 1, .seed = 99, .temperature = 0.5 });
    try testing.expectEqual(@as(usize, 1), selected.token);
    try testing.expectEqual(@as(f32, 1.5), selected.logit);
}

test "llm sampling validates options and logits" {
    const logits = [_]f32{ 1.0, 2.0 };
    const empty = [_]f32{};
    try testing.expectError(error.InvalidArgument, validateOptions(.{ .top_k = 0 }));
    try testing.expectError(error.InvalidArgument, validateOptions(.{ .top_k = top_k_max + 1 }));
    try testing.expectError(error.InvalidArgument, validateOptions(.{ .temperature = 0.0 }));
    try testing.expectError(error.InvalidTokenWindow, argmax(f32, &empty));
    try testing.expectError(error.InvalidTokenWindow, sampleTopK(f32, &empty, .{}));
    try testing.expectError(error.InvalidArgument, sampleTopK(f32, &logits, .{ .temperature = std.math.nan(f64) }));
}
