//! GGUF model diagnostic and generation tool.
//!
//! Loads a GGUF model file, prints metadata and tensor information,
//! and extracts the LlamaConfig. Full text generation with tokenizer
//! integration will come in a later version.
//!
//! Build: zig build generate-gguf
//! Run:   ./zig-out/bin/generate-gguf model.gguf [--max-tokens N]

const std = @import("std");
const zgml = @import("zgml");

const GGUFFile = zgml.gguf.GGUFFile;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    // Parse args via iterator.
    var args_iter: std.process.Args.Iterator = .init(init.minimal.args);
    const prog_name = args_iter.next() orelse "generate-gguf";
    const model_path = args_iter.next() orelse {
        std.debug.print(
            \\Usage: {s} <model.gguf> [--max-tokens N]
            \\
            \\Loads a GGUF model file and prints diagnostic information.
            \\
            \\Options:
            \\  --max-tokens N   Maximum tokens to generate (default: 128)
            \\
        , .{prog_name});
        return;
    };

    var max_tokens: usize = 128;
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--max-tokens")) {
            if (args_iter.next()) |val| {
                max_tokens = std.fmt.parseInt(usize, val, 10) catch {
                    std.debug.print("Error: invalid --max-tokens value: {s}\n", .{val});
                    return;
                };
            }
        }
    }

    // Load GGUF file.
    std.debug.print("Loading {s}...\n", .{model_path});
    var gf = try GGUFFile.open(alloc, io, model_path);
    defer gf.deinit();

    // Print basic info.
    const arch = gf.getMetaString("general.architecture") orelse "unknown";
    const name = gf.getMetaString("general.name") orelse "unnamed";
    std.debug.print("\n=== Model Info ===\n", .{});
    std.debug.print("Name:         {s}\n", .{name});
    std.debug.print("Architecture: {s}\n", .{arch});
    std.debug.print("GGUF version: {d}\n", .{gf.version});
    std.debug.print("Max tokens:   {d}\n", .{max_tokens});

    // Extract config.
    const cfg = zgml.models.gguf_loader.configFromGGUF(&gf);
    std.debug.print("\n=== LlamaConfig ===\n", .{});
    std.debug.print("d_model:     {d}\n", .{cfg.d_model});
    std.debug.print("n_layers:    {d}\n", .{cfg.n_layers});
    std.debug.print("n_heads:     {d}\n", .{cfg.n_heads});
    std.debug.print("n_kv_heads:  {d}\n", .{cfg.n_kv_heads});
    std.debug.print("d_ff:        {d}\n", .{cfg.d_ff});
    std.debug.print("vocab_size:  {d}\n", .{cfg.vocab_size});
    std.debug.print("max_seq_len: {d}\n", .{cfg.max_seq_len});
    std.debug.print("rope_base:   {d:.1}\n", .{cfg.rope_base});

    // Print tensor summary.
    std.debug.print("\n=== Tensors ({d}) ===\n", .{gf.tensors.count()});

    var total_params: u64 = 0;
    var total_bytes: u64 = 0;
    var it = gf.tensors.iterator();
    while (it.next()) |entry| {
        const info = entry.value_ptr.*;
        const n = info.nElems();
        const size = info.dataSize();
        total_params += n;
        total_bytes += size;
    }

    std.debug.print("Total parameters: {d:.2}M\n", .{@as(f64, @floatFromInt(total_params)) / 1e6});
    std.debug.print("Total data size:  {d:.2}MB\n", .{@as(f64, @floatFromInt(total_bytes)) / (1024.0 * 1024.0)});

    // Print individual tensors (sorted iteration not available, so use hash order).
    std.debug.print("\n{s:<45} {s:<10} {s:<20} {s:<12}\n", .{ "Tensor", "Type", "Shape", "Size" });
    std.debug.print("{s}\n", .{"-" ** 90});

    var it2 = gf.tensors.iterator();
    while (it2.next()) |entry| {
        const info = entry.value_ptr.*;
        var shape_buf: [64]u8 = undefined;
        const shape_str = formatShape(&shape_buf, info);
        const size_bytes = info.dataSize();

        std.debug.print("{s:<45} {s:<10} {s:<20} {d:<12}\n", .{
            info.name,
            @tagName(info.type_),
            shape_str,
            size_bytes,
        });
    }

    std.debug.print("\nNote: Full text generation requires tokenizer integration (coming soon).\n", .{});
}

/// Format tensor dimensions as a human-readable shape string like "[4096, 4096]".
fn formatShape(buf: *[64]u8, info: zgml.gguf.TensorInfo) []const u8 {
    var pos: usize = 0;
    buf[pos] = '[';
    pos += 1;

    for (0..info.n_dims) |d| {
        if (d > 0) {
            buf[pos] = ',';
            pos += 1;
            buf[pos] = ' ';
            pos += 1;
        }
        const dim_str = std.fmt.bufPrint(buf[pos..], "{d}", .{info.dims[d]}) catch break;
        pos += dim_str.len;
    }

    buf[pos] = ']';
    pos += 1;
    return buf[0..pos];
}
