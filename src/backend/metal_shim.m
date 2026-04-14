// ObjC implementation of the Metal compute shim.
// Compiled with -fno-objc-arc — callers manage lifetimes via mtl_release().

#import <Metal/Metal.h>
#include "metal_shim.h"

void* mtl_create_device(void) {
    return MTLCreateSystemDefaultDevice();  // +1 retained
}

void* mtl_create_queue(void* device) {
    return [(id<MTLDevice>)device newCommandQueue];
}

void* mtl_create_buffer(void* device, size_t size) {
    return [(id<MTLDevice>)device newBufferWithLength:size
                                             options:MTLResourceStorageModeShared];
}

void* mtl_buffer_contents(void* buffer) {
    return [(id<MTLBuffer>)buffer contents];
}

void* mtl_compile_source(void* device, const char* source, size_t len) {
    NSString* src = [[NSString alloc] initWithBytes:source
                                             length:len
                                           encoding:NSUTF8StringEncoding];
    NSError* error = nil;
    id<MTLLibrary> lib = [(id<MTLDevice>)device newLibraryWithSource:src
                                                             options:nil
                                                               error:&error];
    [src release];
    if (error && !lib) {
        NSLog(@"Metal shader compile error: %@", [error localizedDescription]);
    }
    return lib;
}

void* mtl_create_pipeline(void* device, void* library, const char* name) {
    NSString* fname = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [(id<MTLLibrary>)library newFunctionWithName:fname];
    if (!func) return NULL;

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [(id<MTLDevice>)device newComputePipelineStateWithFunction:func error:&error];
    [func release];
    if (error && !pipeline) {
        NSLog(@"Metal pipeline error: %@", [error localizedDescription]);
    }
    return pipeline;
}

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
    unsigned int threads_y
) {
    id<MTLCommandBuffer> cmd = [(id<MTLCommandQueue>)queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pipeline];

    for (unsigned int i = 0; i < num_buffers; i++) {
        [enc setBuffer:(id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
    }
    if (params && params_size > 0) {
        [enc setBytes:params length:params_size atIndex:params_index];
    }

    MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
    MTLSize group = MTLSizeMake(threads_x, threads_y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void mtl_release(void* obj) {
    [(id)obj release];
}
