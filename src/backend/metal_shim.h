// Thin C wrapper around Metal compute APIs.
// ObjC headers can't be @cImport'd from Zig, so this shim
// exposes the subset we need as plain C functions.

#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#include <stddef.h>

// All Metal objects are opaque void* from Zig's perspective.
// The ObjC implementation casts internally.

// Device + queue
void* mtl_create_device(void);
void* mtl_create_queue(void* device);

// Buffers (shared memory — CPU and GPU see the same pages)
void* mtl_create_buffer(void* device, size_t size);
void* mtl_buffer_contents(void* buffer);

// Shader compilation (from source string at runtime)
void* mtl_compile_source(void* device, const char* source, size_t len);
void* mtl_create_pipeline(void* device, void* library, const char* name);

// Compute dispatch: encode + commit + wait (synchronous).
// Binds `num_buffers` MTLBuffers at indices 0..num_buffers-1,
// then copies `params_size` bytes at index `params_index` via setBytes.
void mtl_dispatch_compute(
    void* queue,
    void* pipeline,
    void** buffers,
    unsigned int num_buffers,
    const void* params,
    size_t params_size,
    unsigned int params_index,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int threads_x,
    unsigned int threads_y);

// Batched command encoding — multiple dispatches share one command buffer.
// mtl_begin_commands creates a command buffer + encoder (returned as opaque handle).
// mtl_encode_dispatch encodes a dispatch into an existing session.
// mtl_commit_and_wait ends encoding, commits, and waits. Releases the session.
void* mtl_begin_commands(void* queue);
void mtl_encode_dispatch(
    void* commands,
    void* pipeline,
    void** buffers,
    unsigned int num_buffers,
    const void* params,
    size_t params_size,
    unsigned int params_index,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int threads_x,
    unsigned int threads_y);
void mtl_commit_and_wait(void* commands);

// Release any Metal object (device, queue, buffer, library, pipeline).
void mtl_release(void* obj);

#endif
