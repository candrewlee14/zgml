//! zgml command-line entry point.

const std = @import("std");
const zgml = @import("zgml");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var args: std.process.Args.Iterator = .init(init.minimal.args);
    const prog = args.next() orelse "zgml";
    const cmd = args.next() orelse {
        usage(prog);
        return;
    };

    if (std.mem.eql(u8, cmd, "inspect")) {
        const path = args.next() orelse {
            usage(prog);
            return;
        };
        try inspectGGUF(alloc, io, path);
        return;
    }

    std.debug.print("unknown command: {s}\n\n", .{cmd});
    usage(prog);
}

fn usage(prog: []const u8) void {
    std.debug.print(
        \\Usage:
        \\  {s} inspect <model.gguf>
        \\
    , .{prog});
}

fn inspectGGUF(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    var gf = try zgml.gguf.GGUFFile.open(alloc, io, path);
    defer gf.deinit();

    const arch = gf.getMetaString("general.architecture") orelse "unknown";
    const name = gf.getMetaString("general.name") orelse "unnamed";
    const cfg = zgml.models.gguf_loader.configFromGGUF(&gf);

    var total_params: u64 = 0;
    var total_bytes: u64 = 0;
    var by_format = [_]u64{0} ** 30;

    var it = gf.tensors.iterator();
    while (it.next()) |entry| {
        const info = entry.value_ptr.*;
        total_params += info.nElems();
        total_bytes += info.dataSize();
        by_format[@intFromEnum(info.type_)] += 1;
    }

    std.debug.print("name: {s}\n", .{name});
    std.debug.print("architecture: {s}\n", .{arch});
    std.debug.print("gguf_version: {d}\n", .{gf.version});
    std.debug.print("tensors: {d}\n", .{gf.tensors.count()});
    std.debug.print("parameters_m: {d:.2}\n", .{@as(f64, @floatFromInt(total_params)) / 1e6});
    std.debug.print("data_mb: {d:.2}\n", .{@as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0)});
    std.debug.print("config: d_model={d} layers={d} heads={d} kv_heads={d} d_ff={d} context={d} vocab={d} rope_base={d:.1}\n", .{
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.d_ff,
        cfg.max_seq_len,
        cfg.vocab_size,
        cfg.rope_base,
    });

    std.debug.print("formats:", .{});
    inline for (std.meta.fields(zgml.gguf.GGMLType)) |field| {
        const tag: zgml.gguf.GGMLType = @enumFromInt(field.value);
        const count = by_format[@intFromEnum(tag)];
        if (count != 0) std.debug.print(" {s}={d}", .{ field.name, count });
    }
    std.debug.print("\n", .{});
}
