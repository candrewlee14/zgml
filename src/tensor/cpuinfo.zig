//! Portable runtime CPU feature detection.
//!
//! Detects SIMD capabilities at runtime to select optimal kernel widths.
//! Returns the widest usable SIMD register width in bytes.

const std = @import("std");
const builtin = @import("builtin");

/// Detected SIMD width in bytes (16, 32, or 64).
pub const SimdWidth = enum(u8) {
    sse = 16,
    avx = 32,
    avx512 = 64,
};

/// Detect the widest SIMD register width available at runtime.
/// Falls back to 32 bytes (AVX/NEON-equivalent) if detection is unavailable.
pub fn detectSimdWidth() SimdWidth {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => detectX86(),
        .aarch64 => detectAarch64(),
        else => .avx, // conservative default
    };
}

/// Optimal SIMD vector lane count for a given element type, based on runtime detection.
pub fn optimalVecSize(comptime T: type) usize {
    const width = detectSimdWidth();
    const lanes = @as(usize, @intFromEnum(width)) / @sizeOf(T);
    return if (lanes >= 4) lanes else 4;
}

// ---------------------------------------------------------------------------
// x86/x86_64: CPUID-based detection
// ---------------------------------------------------------------------------

/// x86-specific helpers are only defined when compiling for x86/x86_64.
/// On other architectures these are unreachable stubs so the inline asm
/// is never analyzed by the compiler.
const is_x86 = builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86;

const CpuidResult = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32, subleaf: u32) CpuidResult {
    if (comptime !is_x86) unreachable;
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (subleaf),
    );
    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn getXcr0() u32 {
    if (comptime !is_x86) unreachable;
    var eax: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("xgetbv"
        : [eax_out] "={eax}" (eax),
          [edx_out] "={edx}" (edx),
        : [ecx_in] "{ecx}" (@as(u32, 0)),
    );
    edx = edx; // discard edx output
    return eax;
}

fn detectX86() SimdWidth {
    if (comptime !is_x86) return .sse; // conservative fallback
    const leaf1 = cpuid(1, 0);

    // Check OSXSAVE (ECX bit 27) — required for AVX/AVX-512 state saving
    const has_osxsave = (leaf1.ecx >> 27) & 1 != 0;
    if (!has_osxsave) return .sse;

    const xcr0 = getXcr0();
    // AVX requires XCR0 bits 1 (SSE) and 2 (AVX)
    const has_avx_save = (xcr0 & 0x6) == 0x6;
    if (!has_avx_save) return .sse;

    // Check AVX2 (leaf 7, EBX bit 5)
    const leaf7 = cpuid(7, 0);
    const has_avx2 = (leaf7.ebx >> 5) & 1 != 0;
    if (!has_avx2) return .sse;

    // Check AVX-512F (leaf 7, EBX bit 16) + OS support for ZMM (XCR0 bits 5,6,7)
    const has_avx512f = (leaf7.ebx >> 16) & 1 != 0;
    const has_avx512_save = (xcr0 & 0xE0) == 0xE0; // opmask + zmm_hi256 + hi16_zmm
    if (has_avx512f and has_avx512_save) return .avx512;

    return .avx;
}

// ---------------------------------------------------------------------------
// AArch64: NEON is always 128-bit, SVE varies
// ---------------------------------------------------------------------------

fn detectAarch64() SimdWidth {
    // NEON (128-bit) is mandatory on AArch64.
    // SVE detection would require reading system registers which needs kernel support.
    // For now, report 128-bit. SVE support can be added later.
    return .sse; // 128-bit = 16 bytes
}
