const std = @import("std");
const zgml = @import("zgml");

const Tensor = zgml.Tensor;
const ComputeGraph = zgml.ComputeGraph;
const loss = zgml.loss;
const nn = zgml.nn;

fn maxAbsDiff(actual: []const f32, expected: []const f32) f32 {
    std.debug.assert(actual.len == expected.len);
    var max_diff: f32 = 0;
    for (actual, expected) |a, e| {
        const diff = @abs(a - e);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

fn expectAllClose(actual: []const f32, expected: []const f32, tol: f32) !void {
    std.debug.assert(actual.len == expected.len);
    for (actual, expected, 0..) |a, e, i| {
        _ = i;
        try std.testing.expectApproxEqAbs(e, a, tol);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const backing = gpa.allocator();

    var g = ComputeGraph(f32).init(backing);
    defer g.deinit();
    const a = g.allocator();

    const ks: usize = 3;
    const n_filters: usize = 2;
    const n_classes: usize = 3;
    const batch_size: usize = 2;
    const in_w: usize = 4;
    const in_h: usize = 4;
    const conv_w = in_w - ks + 1;
    const conv_h = in_h - ks + 1;
    const flat_dim = (conv_w / 2) * (conv_h / 2) * n_filters;

    const conv_k_ref = [_]f32{
        0.1523585468530655,   -0.5199920535087585,  0.37522560358047485,
        0.47028234601020813,  -0.9755175709724426,  -0.6510897278785706,
        0.06392019987106323,  -0.15812130272388458, -0.00840057898312807,
        -0.4265219569206238,  0.4396989941596985,   0.3888959586620331,
        0.03301534801721573,  0.5636206269264221,   0.23375466465950012,
        -0.42964622378349304, 0.18437539041042328,  -0.4794413149356842,
    };
    const fc_w_ref = [_]f32{
        0.43922513723373413, -0.024962956085801125, -0.09243118017911911,
        -0.3404647707939148, 0.6112706661224365,    -0.07726474106311798,
    };
    const xs_ref = [_]f32{
        -0.4283278286457062,  -0.35213354229927063, 0.5323091745376587,   0.3654440641403198,
        0.4127326011657715,   0.4308210015296936,   2.1416475772857666,   -0.40641501545906067,
        -0.5122427344322205,  -0.8137727379798889,  0.6159794330596924,   1.1289722919464111,
        -0.11394745856523514, -0.8401564955711365,  -0.824481189250946,   0.6505928039550781,
        0.7432541847229004,   0.543154239654541,    -0.6655097007751465,  0.23216132819652557,
        0.11668580770492554,  0.21868859231472015,  0.8714287877082825,   0.2235955446958542,
        0.6789135336875916,   0.06757906824350357,  0.2891193926334381,   0.6312882304191589,
        -1.4571558237075806,  -0.3196712136268616,  -0.47037264704704285, -0.6388778686523438,
    };
    const ys_ref = [_]f32{ 0.0, 2.0 };

    const pooled_expected = [_]f32{ 0.9268430, 1.5745969, 0.25751454, 1.3060079 };
    const logits_expected = [_]f32{ -0.12900202, 0.9393681, -0.20733, -0.33154282, 0.791896, -0.12471073 };
    const loss_expected: f32 = 1.5188972950;

    const fc_w_grad_expected = [_]f32{
        -0.3433099687099457, 0.3535996377468109, -0.01028965413570404,
        -0.5013872981071472, 0.8524644374847412, -0.3510770797729492,
    };
    const fc_b_grad_expected = [_]f32{ -0.30234625935554504, 0.5908272862434387, -0.2884809970855713 };
    const conv_k_grad_expected = [_]f32{
        -0.07051549851894379, -0.06685634702444077, -0.34737417101860046,
        0.1449003666639328,   0.15976813435554504,  -0.09723096340894699,
        -0.07979748398065567, 0.13781847059726715,  0.12433333694934845,
        -0.08933824300765991, 0.20409604907035828,  0.2661745846271515,
        0.252902090549469,    0.6791850328445435,   -0.07611781358718872,
        -0.5082465410232544,  0.13608230650424957,  0.26960235834121704,
    };
    const conv_b_grad_expected = [_]f32{ -0.12088222801685333, 0.4863830506801605 };

    const conv_k = try g.param(&.{ ks, ks, 1, n_filters });
    const conv_b = try g.param(&.{n_filters});
    const fc_w = try g.param(&.{ n_classes, flat_dim });
    const fc_b = try g.param(&.{n_classes});
    conv_k.setData(&conv_k_ref);
    conv_b.setData(&.{ 0.0, 0.0 });
    fc_w.setData(&fc_w_ref);
    fc_b.setData(&.{ 0.0, 0.0, 0.0 });

    const xs = try Tensor(f32).init(a, &.{ in_w, in_h, 1, batch_size });
    const ys = try Tensor(f32).init(a, &.{batch_size});
    xs.setData(&xs_ref);
    ys.setData(&ys_ref);

    const conv_out = xs.conv2d(conv_k);
    const cb_4d = conv_b.reshape(&.{ 1, 1, n_filters, 1 });
    const conv_act = conv_out.add(cb_4d.repeat(conv_out.ne[0..conv_out.n_dims])).relu();
    const pooled = conv_act.maxPool2d();
    const flat = pooled.reshape(&.{ flat_dim, batch_size });
    const logits = nn.linear(f32, flat, fc_w, fc_b);
    const ce = loss.crossEntropy(f32, logits, ys);

    var g_unfused = ComputeGraph(f32).init(backing);
    defer g_unfused.deinit();
    const au = g_unfused.allocator();

    const conv_k_u = try g_unfused.param(&.{ ks, ks, 1, n_filters });
    const conv_b_u = try g_unfused.param(&.{n_filters});
    const fc_w_u = try g_unfused.param(&.{ n_classes, flat_dim });
    const fc_b_u = try g_unfused.param(&.{n_classes});
    conv_k_u.setData(&conv_k_ref);
    conv_b_u.setData(&.{ 0.0, 0.0 });
    fc_w_u.setData(&fc_w_ref);
    fc_b_u.setData(&.{ 0.0, 0.0, 0.0 });

    const xs_u = try Tensor(f32).init(au, &.{ in_w, in_h, 1, batch_size });
    const ys_u = try Tensor(f32).init(au, &.{batch_size});
    xs_u.setData(&xs_ref);
    ys_u.setData(&ys_ref);

    const conv_out_u = xs_u.conv2d(conv_k_u);
    const cb_4d_u = conv_b_u.reshape(&.{ 1, 1, n_filters, 1 });
    const conv_act_u = conv_out_u.add(cb_4d_u.repeat(conv_out_u.ne[0..conv_out_u.n_dims])).relu();
    const pooled_u = conv_act_u.maxPool2d();
    const flat_u = pooled_u.reshape(&.{ flat_dim, batch_size });
    const logits_u = nn.linear(f32, flat_u, fc_w_u, fc_b_u);
    const ce_u = loss.crossEntropy(f32, logits_u, ys_u);

    try g.buildForward(ce);
    try g.buildBackward(false);
    try g.fusionPass();
    _ = ce.grad.?.setAllScalar(1);

    try g_unfused.buildForward(ce_u);
    try g_unfused.buildBackward(false);
    _ = ce_u.grad.?.setAllScalar(1);

    g.compute();
    g_unfused.compute();

    const tol: f32 = 1e-5;
    try expectAllClose(pooled.data, &pooled_expected, tol);
    try expectAllClose(logits.data, &logits_expected, tol);
    try std.testing.expectApproxEqAbs(loss_expected, ce.data[0], tol);
    try expectAllClose(fc_w.grad.?.data, &fc_w_grad_expected, tol);
    try expectAllClose(fc_b.grad.?.data, &fc_b_grad_expected, tol);
    try expectAllClose(conv_k.grad.?.data, &conv_k_grad_expected, tol);
    try expectAllClose(conv_b.grad.?.data, &conv_b_grad_expected, tol);
    try expectAllClose(pooled.data, pooled_u.data, tol);
    try expectAllClose(logits.data, logits_u.data, tol);
    try std.testing.expectApproxEqAbs(ce_u.data[0], ce.data[0], tol);
    try expectAllClose(conv_k_u.grad.?.data, conv_k.grad.?.data, tol);
    try expectAllClose(conv_b_u.grad.?.data, conv_b.grad.?.data, tol);
    try expectAllClose(fc_w_u.grad.?.data, fc_w.grad.?.data, tol);
    try expectAllClose(fc_b_u.grad.?.data, fc_b.grad.?.data, tol);

    const stdout_file = std.fs.File.stdout();
    var buf: [2048]u8 = undefined;
    var w = stdout_file.writer(&buf);

    try w.interface.print("grad-check passed\n", .{});
    try w.interface.print("  loss diff: {d:.10}\n", .{@abs(ce.data[0] - loss_expected)});
    try w.interface.print("  conv_out materialized in unfused path only\n", .{});
    try w.interface.print("  pooled max diff: {d:.10}\n", .{maxAbsDiff(pooled.data, &pooled_expected)});
    try w.interface.print("  logits max diff: {d:.10}\n", .{maxAbsDiff(logits.data, &logits_expected)});
    try w.interface.print("  conv_k.grad max diff: {d:.10}\n", .{maxAbsDiff(conv_k.grad.?.data, &conv_k_grad_expected)});
    try w.interface.print("  conv_b.grad max diff: {d:.10}\n", .{maxAbsDiff(conv_b.grad.?.data, &conv_b_grad_expected)});
    try w.interface.print("  fc_w.grad max diff: {d:.10}\n", .{maxAbsDiff(fc_w.grad.?.data, &fc_w_grad_expected)});
    try w.interface.print("  fc_b.grad max diff: {d:.10}\n", .{maxAbsDiff(fc_b.grad.?.data, &fc_b_grad_expected)});
    w.interface.flush() catch {};
}
