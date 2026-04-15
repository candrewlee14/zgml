//! MNIST IDX file loading.
//!
//! Loads MNIST image and label files in the standard IDX binary format
//! into flat f32 slices suitable for use with DataLoader.
//!
//! Usage:
//! ```
//! var ds = try MnistDataset.load(alloc, "data/train-images-idx3-ubyte",
//!                                       "data/train-labels-idx1-ubyte");
//! defer ds.deinit(alloc);
//! const ppi = ds.pixelsPerImage(); // 784 for 28x28
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;
const file_compat = @import("../file_compat.zig");

fn readU32Big(buf: []const u8) u32 {
    return std.mem.readInt(u32, buf[0..4], .big);
}

pub const MnistDataset = struct {
    images: []f32,
    labels: []f32,
    n_samples: usize,
    rows: usize,
    cols: usize,

    /// Load MNIST images and labels from IDX files.
    ///
    /// Images are normalized to [0, 1] (divided by 255).
    /// Labels are stored as f32 class indices (e.g., 0.0, 1.0, ..., 9.0).
    pub fn load(alloc: Alloc, images_path: []const u8, labels_path: []const u8) !MnistDataset {
        // -- Load images --
        const img_raw = try file_compat.readToEndAlloc(alloc, images_path, 128 * 1024 * 1024);
        defer alloc.free(img_raw);

        const img_magic = readU32Big(img_raw[0..]);
        if (img_magic != 2051) return error.BadImageMagic;
        const n_images = readU32Big(img_raw[4..]);
        const rows = readU32Big(img_raw[8..]);
        const cols = readU32Big(img_raw[12..]);
        const pixels = img_raw[16..];

        const total_px = n_images * rows * cols;
        const images = try alloc.alloc(f32, total_px);
        errdefer alloc.free(images);
        for (0..total_px) |i| {
            images[i] = @as(f32, @floatFromInt(pixels[i])) / 255.0;
        }

        // -- Load labels --
        const lbl_raw = try file_compat.readToEndAlloc(alloc, labels_path, 16 * 1024 * 1024);
        defer alloc.free(lbl_raw);

        const lbl_magic = readU32Big(lbl_raw[0..]);
        if (lbl_magic != 2049) return error.BadLabelMagic;
        const n_labels = readU32Big(lbl_raw[4..]);
        const lbl_bytes = lbl_raw[8..];

        if (n_images != n_labels) return error.SampleCountMismatch;

        const labels = try alloc.alloc(f32, n_labels);
        for (0..n_labels) |i| {
            labels[i] = @floatFromInt(lbl_bytes[i]);
        }

        return .{
            .images = images,
            .labels = labels,
            .n_samples = n_images,
            .rows = rows,
            .cols = cols,
        };
    }

    pub fn deinit(self: *MnistDataset, alloc: Alloc) void {
        alloc.free(self.images);
        alloc.free(self.labels);
    }

    /// Number of pixels per single image (rows * cols).
    pub fn pixelsPerImage(self: *const MnistDataset) usize {
        return self.rows * self.cols;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "readU32Big - image magic" {
    const buf = [_]u8{ 0x00, 0x00, 0x08, 0x03, 0xAA, 0xBB };
    try testing.expectEqual(@as(u32, 2051), readU32Big(&buf));
}

test "readU32Big - label magic" {
    const buf = [_]u8{ 0x00, 0x00, 0x08, 0x01 };
    try testing.expectEqual(@as(u32, 2049), readU32Big(&buf));
}

test "MnistDataset - pixelsPerImage" {
    var ds = MnistDataset{
        .images = &.{},
        .labels = &.{},
        .n_samples = 0,
        .rows = 28,
        .cols = 28,
    };
    try testing.expectEqual(@as(usize, 784), ds.pixelsPerImage());
}
