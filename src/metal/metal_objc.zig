//! Objective-C runtime bridge for Metal GPU compute.
//!
//! Provides typed wrappers around `objc_msgSend` for the subset of the
//! Metal API needed by the zgml GPU backend: device creation, buffer
//! allocation, shader compilation, command encoding, dispatch, and
//! synchronization.
//!
//! All Metal objects are represented as opaque `id` pointers (`?*anyopaque`).
//! We use `@ptrCast` to pass them through the ObjC runtime without importing
//! any Objective-C headers beyond `<objc/message.h>`.

const std = @import("std");

const c = @cImport({
    @cInclude("objc/message.h");
    @cInclude("objc/runtime.h");
});

// ---------------------------------------------------------------------------
// Low-level ObjC runtime helpers
// ---------------------------------------------------------------------------

pub const id = ?*anyopaque;
pub const SEL = c.SEL;
pub const Class = c.Class;

fn sel(name: [*:0]const u8) SEL {
    return c.sel_registerName(name);
}

fn cls(name: [*:0]const u8) Class {
    return c.objc_getClass(name);
}

/// Untyped objc_msgSend — callers cast the result.
fn msgSend(target: id, selector: SEL) id {
    const func: *const fn (id, SEL) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(target, selector);
}

fn msgSend1(target: id, selector: SEL, arg0: id) id {
    const func: *const fn (id, SEL, id) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(target, selector, arg0);
}

fn msgSend2(target: id, selector: SEL, arg0: id, arg1: id) id {
    const func: *const fn (id, SEL, id, id) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(target, selector, arg0, arg1);
}

fn msgSendUsize(target: id, selector: SEL) usize {
    const func: *const fn (id, SEL) callconv(.C) usize = @ptrCast(&c.objc_msgSend);
    return func(target, selector);
}

fn msgSendPtr(target: id, selector: SEL) [*]u8 {
    const func: *const fn (id, SEL) callconv(.C) [*]u8 = @ptrCast(&c.objc_msgSend);
    return func(target, selector);
}

// ---------------------------------------------------------------------------
// NSString helpers
// ---------------------------------------------------------------------------

fn nsString(str: [*:0]const u8) id {
    const NSString = cls("NSString");
    const alloc_sel = sel("alloc");
    const init_sel = sel("initWithUTF8String:");
    const raw = msgSend(@ptrCast(NSString), alloc_sel);
    const func: *const fn (id, SEL, [*:0]const u8) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(raw, init_sel, str);
}

fn nsStringFromSlice(alloc: std.mem.Allocator, str: []const u8) !id {
    const z = try alloc.dupeZ(u8, str);
    defer alloc.free(z);
    return nsString(z.ptr);
}

// ---------------------------------------------------------------------------
// Metal Device
// ---------------------------------------------------------------------------

/// `MTLCreateSystemDefaultDevice()` — returns the default GPU.
pub fn createSystemDefaultDevice() id {
    // This is a C function, not an ObjC message.
    const func = @extern(*const fn () callconv(.C) id, .{ .name = "MTLCreateSystemDefaultDevice" });
    return func();
}

/// `[device name]` — GPU name string (for diagnostics).
pub fn deviceName(device: id) id {
    return msgSend(device, sel("name"));
}

// ---------------------------------------------------------------------------
// Metal Command Queue
// ---------------------------------------------------------------------------

/// `[device newCommandQueue]`
pub fn newCommandQueue(device: id) id {
    return msgSend(device, sel("newCommandQueue"));
}

// ---------------------------------------------------------------------------
// Metal Buffers
// ---------------------------------------------------------------------------

/// MTLResourceStorageModeShared = 0 (shared CPU/GPU on Apple Silicon).
const MTLResourceStorageModeShared: usize = 0;

/// `[device newBufferWithLength:options:]`
pub fn newBuffer(device: id, length: usize) id {
    const func: *const fn (id, SEL, usize, usize) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(device, sel("newBufferWithLength:options:"), length, MTLResourceStorageModeShared);
}

/// `[device newBufferWithBytes:length:options:]`
pub fn newBufferWithBytes(device: id, bytes: [*]const u8, length: usize) id {
    const func: *const fn (id, SEL, [*]const u8, usize, usize) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    return func(device, sel("newBufferWithBytes:length:options:"), bytes, length, MTLResourceStorageModeShared);
}

/// `[buffer contents]` — returns CPU-visible pointer to buffer data.
pub fn bufferContents(buffer: id) [*]u8 {
    return msgSendPtr(buffer, sel("contents"));
}

/// `[buffer length]`
pub fn bufferLength(buffer: id) usize {
    return msgSendUsize(buffer, sel("length"));
}

// ---------------------------------------------------------------------------
// Metal Library (shader compilation)
// ---------------------------------------------------------------------------

/// `[device newLibraryWithSource:options:error:]`
pub fn newLibraryWithSource(device: id, source: id) id {
    const func: *const fn (id, SEL, id, id, *id) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    var err: id = null;
    const lib = func(device, sel("newLibraryWithSource:options:error:"), source, null, &err);
    if (lib == null and err != null) {
        // Log error description for debugging.
        const desc = msgSend(err, sel("localizedDescription"));
        const utf8_sel = sel("UTF8String");
        const utf8_func: *const fn (id, SEL) callconv(.C) [*:0]const u8 = @ptrCast(&c.objc_msgSend);
        const msg = utf8_func(desc, utf8_sel);
        std.log.err("Metal shader compilation failed: {s}", .{msg});
    }
    return lib;
}

/// `[library newFunctionWithName:]`
pub fn newFunctionWithName(library: id, name: id) id {
    return msgSend1(library, sel("newFunctionWithName:"), name);
}

// ---------------------------------------------------------------------------
// Metal Compute Pipeline
// ---------------------------------------------------------------------------

/// `[device newComputePipelineStateWithFunction:error:]`
pub fn newComputePipelineState(device: id, function: id) id {
    const func: *const fn (id, SEL, id, *id) callconv(.C) id = @ptrCast(&c.objc_msgSend);
    var err: id = null;
    const pso = func(device, sel("newComputePipelineStateWithFunction:error:"), function, &err);
    if (pso == null and err != null) {
        std.log.err("Metal pipeline creation failed", .{});
    }
    return pso;
}

/// `[pso maxTotalThreadsPerThreadgroup]`
pub fn maxTotalThreadsPerThreadgroup(pso: id) usize {
    return msgSendUsize(pso, sel("maxTotalThreadsPerThreadgroup"));
}

// ---------------------------------------------------------------------------
// Metal Command Buffer & Compute Encoder
// ---------------------------------------------------------------------------

/// `[queue commandBuffer]`
pub fn newCommandBuffer(queue: id) id {
    return msgSend(queue, sel("commandBuffer"));
}

/// `[cmdBuf computeCommandEncoder]`
pub fn newComputeEncoder(cmd_buf: id) id {
    return msgSend(cmd_buf, sel("computeCommandEncoder"));
}

/// `[encoder setComputePipelineState:]`
pub fn setComputePipelineState(encoder: id, pso: id) void {
    _ = msgSend1(encoder, sel("setComputePipelineState:"), pso);
}

/// MTLSize struct for dispatch.
pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,
};

/// `[encoder setBuffer:offset:atIndex:]`
pub fn setBuffer(encoder: id, buffer: id, offset: usize, index: usize) void {
    const func: *const fn (id, SEL, id, usize, usize) callconv(.C) void = @ptrCast(&c.objc_msgSend);
    func(encoder, sel("setBuffer:offset:atIndex:"), buffer, offset, index);
}

/// `[encoder setBytes:length:atIndex:]`
pub fn setBytes(encoder: id, bytes: [*]const u8, length: usize, index: usize) void {
    const func: *const fn (id, SEL, [*]const u8, usize, usize) callconv(.C) void = @ptrCast(&c.objc_msgSend);
    func(encoder, sel("setBytes:length:atIndex:"), bytes, length, index);
}

/// `[encoder dispatchThreadgroups:threadsPerThreadgroup:]`
pub fn dispatchThreadgroups(encoder: id, grid_size: MTLSize, group_size: MTLSize) void {
    const func: *const fn (id, SEL, MTLSize, MTLSize) callconv(.C) void = @ptrCast(&c.objc_msgSend);
    func(encoder, sel("dispatchThreadgroups:threadsPerThreadgroup:"), grid_size, group_size);
}

/// `[encoder endEncoding]`
pub fn endEncoding(encoder: id) void {
    _ = msgSend(encoder, sel("endEncoding"));
}

/// `[cmdBuf commit]`
pub fn commit(cmd_buf: id) void {
    _ = msgSend(cmd_buf, sel("commit"));
}

/// `[cmdBuf waitUntilCompleted]`
pub fn waitUntilCompleted(cmd_buf: id) void {
    _ = msgSend(cmd_buf, sel("waitUntilCompleted"));
}

/// `[obj release]`
pub fn release(obj: id) void {
    _ = msgSend(obj, sel("release"));
}
