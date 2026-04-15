//! GPT-2 byte-level BPE tokenizer.
//!
//! Implements encode (text → token IDs) and decode (token IDs → text) using
//! the GPT-2 tokenization scheme: UTF-8 bytes are mapped to printable Unicode
//! characters, then BPE merges are applied iteratively.
//!
//! ```
//! var tok = try GPT2Tokenizer.init(alloc, "vocab.json", "merges.txt");
//! defer tok.deinit();
//! const ids = try tok.encode(alloc, "Once upon a time");
//! defer alloc.free(ids);
//! const text = try tok.decode(alloc, ids);
//! defer alloc.free(text);
//! ```

const std = @import("std");
const Alloc = std.mem.Allocator;
const file_compat = @import("file_compat.zig");

pub const GPT2Tokenizer = struct {
    alloc: Alloc,
    /// Token string → ID
    encoder: std.StringHashMapUnmanaged(u32) = .empty,
    /// ID → token string (borrowed from encoder keys)
    decoder: std.AutoHashMapUnmanaged(u32, []const u8) = .empty,
    /// Ordered BPE merge rules: "a b" → priority (lower = higher priority)
    merges: std.StringHashMapUnmanaged(u32) = .empty,
    /// Storage for merge key strings
    merge_keys: std.ArrayList([]u8) = .empty,
    /// Allocated vocab token key strings
    vocab_keys: std.ArrayList([]u8) = .empty,
    /// Raw vocab.json file data
    vocab_data: []u8 = &.{},

    pub fn init(alloc: Alloc, vocab_path: []const u8, merges_path: []const u8) !GPT2Tokenizer {
        var self = GPT2Tokenizer{ .alloc = alloc };
        errdefer self.deinit();

        // Load vocab.json
        self.vocab_data = try readFile(alloc, vocab_path);
        try parseVocabJson(&self);

        // Load merges.txt
        const merges_data = try readFile(alloc, merges_path);
        defer alloc.free(merges_data);
        try parseMerges(&self, merges_data);

        return self;
    }

    pub fn deinit(self: *GPT2Tokenizer) void {
        const alloc = self.alloc;
        self.encoder.deinit(alloc);
        self.decoder.deinit(alloc);
        self.merges.deinit(alloc);
        for (self.merge_keys.items) |key| alloc.free(key);
        self.merge_keys.deinit(alloc);
        for (self.vocab_keys.items) |key| alloc.free(key);
        self.vocab_keys.deinit(alloc);
        if (self.vocab_data.len > 0) alloc.free(self.vocab_data);
    }

    /// Decode token IDs back to UTF-8 text.
    pub fn decode(self: *const GPT2Tokenizer, alloc: Alloc, ids: []const u32) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        errdefer result.deinit(alloc);

        for (ids) |id| {
            const token_str = self.decoder.get(id) orelse continue;
            // Reverse the GPT-2 byte encoding: each UTF-8 codepoint maps to one byte
            var i: usize = 0;
            const bytes = token_str;
            while (i < bytes.len) {
                const cp_len = std.unicode.utf8ByteSequenceLength(bytes[i]) catch 1;
                if (i + cp_len > bytes.len) break;
                const cp = std.unicode.utf8Decode(bytes[i..][0..cp_len]) catch {
                    try result.append(alloc, bytes[i]);
                    i += 1;
                    continue;
                };
                const byte = unicodeToByteTable(cp);
                try result.append(alloc, byte);
                i += cp_len;
            }
        }

        return result.toOwnedSlice(alloc);
    }

    /// Encode UTF-8 text to token IDs using byte-level BPE.
    pub fn encode(self: *const GPT2Tokenizer, alloc: Alloc, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(alloc);

        // Pre-tokenize: split on whitespace boundaries (GPT-2 style)
        var word_start: usize = 0;
        while (word_start < text.len) {
            var word_end = word_start + 1;
            while (word_end < text.len) {
                if (text[word_end] == ' ' and word_end > word_start) break;
                word_end += 1;
            }

            const word = text[word_start..word_end];

            // Convert bytes to GPT-2 Unicode representation
            var symbols: std.ArrayList([]const u8) = .empty;
            defer symbols.deinit(alloc);

            for (word) |byte| {
                const cp = byteToUnicodeTable(byte);
                var buf: [4]u8 = undefined;
                const len = std.unicode.utf8Encode(cp, &buf) catch continue;
                const sym = try alloc.dupe(u8, buf[0..len]);
                try symbols.append(alloc, sym);
            }
            defer for (symbols.items) |s| alloc.free(s);

            // Apply BPE merges
            try applyBPE(self, alloc, &symbols);

            // Look up each merged symbol in the vocab
            for (symbols.items) |sym| {
                if (self.encoder.get(sym)) |id| {
                    try ids.append(alloc, id);
                }
            }

            word_start = word_end;
        }

        return ids.toOwnedSlice(alloc);
    }

    fn applyBPE(self: *const GPT2Tokenizer, _: Alloc, symbols: *std.ArrayList([]const u8)) !void {
        while (symbols.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;

            var merge_key_buf: [256]u8 = undefined;
            for (0..symbols.items.len - 1) |i| {
                const a = symbols.items[i];
                const b = symbols.items[i + 1];
                if (a.len + 1 + b.len > merge_key_buf.len) continue;

                var pos: usize = 0;
                @memcpy(merge_key_buf[pos..][0..a.len], a);
                pos += a.len;
                merge_key_buf[pos] = ' ';
                pos += 1;
                @memcpy(merge_key_buf[pos..][0..b.len], b);
                pos += b.len;

                if (self.merges.get(merge_key_buf[0..pos])) |priority| {
                    if (priority < best_priority) {
                        best_priority = priority;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == null) break;

            const idx = best_idx.?;
            const a = symbols.items[idx];
            const b = symbols.items[idx + 1];
            const merged = self.alloc.alloc(u8, a.len + b.len) catch break;
            @memcpy(merged[0..a.len], a);
            @memcpy(merged[a.len..], b);

            self.alloc.free(a);
            self.alloc.free(b);
            symbols.items[idx] = merged;
            _ = symbols.orderedRemove(idx + 1);
        }
    }
};

/// GPT-2 byte → Unicode mapping.
fn byteToUnicodeTable(byte: u8) u21 {
    return switch (byte) {
        '!'...'~' => byte,
        0xA1...0xAC => byte,
        0xAE...0xFF => byte,
        else => blk: {
            var offset: u21 = 0;
            for (0..byte) |b| {
                const in_range = (b >= '!' and b <= '~') or
                    (b >= 0xA1 and b <= 0xAC) or
                    (b >= 0xAE and b <= 0xFF);
                if (!in_range) offset += 1;
            }
            break :blk 256 + offset;
        },
    };
}

/// Reverse mapping: Unicode code point → byte.
fn unicodeToByteTable(cp: u21) u8 {
    if (cp >= '!' and cp <= '~') return @intCast(cp);
    if (cp >= 0xA1 and cp <= 0xAC) return @intCast(cp);
    if (cp >= 0xAE and cp <= 0xFF) return @intCast(cp);

    if (cp >= 256) {
        const target_offset: u21 = cp - 256;
        var offset: u21 = 0;
        for (0..256) |b| {
            const in_range = (b >= '!' and b <= '~') or
                (b >= 0xA1 and b <= 0xAC) or
                (b >= 0xAE and b <= 0xFF);
            if (!in_range) {
                if (offset == target_offset) return @intCast(b);
                offset += 1;
            }
        }
    }
    return '?';
}

fn readFile(alloc: Alloc, path: []const u8) ![]u8 {
    return file_compat.readToEndAlloc(alloc, path, std.math.maxInt(usize));
}

/// Parse a minimal vocab.json: {"token": id, "token2": id2, ...}
fn parseVocabJson(self: *GPT2Tokenizer) !void {
    const alloc = self.alloc;
    const json = self.vocab_data;
    var pos: usize = 0;

    while (pos < json.len and json[pos] != '{') : (pos += 1) {}
    pos += 1;

    while (pos < json.len) {
        while (pos < json.len and (json[pos] == ' ' or json[pos] == '\n' or
            json[pos] == '\r' or json[pos] == '\t' or json[pos] == ','))
        {
            pos += 1;
        }
        if (pos >= json.len or json[pos] == '}') break;

        if (json[pos] != '"') {
            pos += 1;
            continue;
        }
        pos += 1;
        const key_start = pos;
        while (pos < json.len) {
            if (json[pos] == '\\') {
                pos += 2;
                continue;
            }
            if (json[pos] == '"') break;
            pos += 1;
        }
        const raw_key = json[key_start..pos];
        pos += 1;

        const key = try unescapeJsonString(alloc, raw_key);
        try self.vocab_keys.append(alloc, key);

        while (pos < json.len and json[pos] != ':') : (pos += 1) {}
        pos += 1;

        while (pos < json.len and (json[pos] == ' ' or json[pos] == '\n' or
            json[pos] == '\r' or json[pos] == '\t'))
        {
            pos += 1;
        }

        const val_start = pos;
        while (pos < json.len and (json[pos] >= '0' and json[pos] <= '9')) : (pos += 1) {}
        const id = std.fmt.parseInt(u32, json[val_start..pos], 10) catch continue;

        try self.encoder.put(alloc, key, id);
        try self.decoder.put(alloc, id, key);
    }
}

/// Parse merges.txt: first line is header, then "token_a token_b" per line.
fn parseMerges(self: *GPT2Tokenizer, data: []const u8) !void {
    const alloc = self.alloc;
    var line_iter = std.mem.splitScalar(u8, data, '\n');
    var priority: u32 = 0;

    _ = line_iter.next(); // skip header

    while (line_iter.next()) |line| {
        if (line.len == 0) continue;
        if (line[0] == '#') continue;

        const key = try alloc.dupe(u8, line);
        try self.merge_keys.append(alloc, key);
        try self.merges.put(alloc, key, priority);
        priority += 1;
    }
}

/// Unescape JSON string: handle \uXXXX, \\, \", \n, \t, \r, \/, \b, \f
fn unescapeJsonString(alloc: Alloc, raw: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(alloc);

    var i: usize = 0;
    while (i < raw.len) {
        if (raw[i] == '\\' and i + 1 < raw.len) {
            switch (raw[i + 1]) {
                '\\' => {
                    try result.append(alloc, '\\');
                    i += 2;
                },
                '"' => {
                    try result.append(alloc, '"');
                    i += 2;
                },
                'n' => {
                    try result.append(alloc, '\n');
                    i += 2;
                },
                't' => {
                    try result.append(alloc, '\t');
                    i += 2;
                },
                'r' => {
                    try result.append(alloc, '\r');
                    i += 2;
                },
                '/' => {
                    try result.append(alloc, '/');
                    i += 2;
                },
                'b' => {
                    try result.append(alloc, 0x08);
                    i += 2;
                },
                'f' => {
                    try result.append(alloc, 0x0C);
                    i += 2;
                },
                'u' => {
                    if (i + 5 < raw.len) {
                        const hex = raw[i + 2 .. i + 6];
                        const cp = std.fmt.parseInt(u21, hex, 16) catch {
                            try result.append(alloc, '\\');
                            i += 1;
                            continue;
                        };
                        var buf: [4]u8 = undefined;
                        const len = std.unicode.utf8Encode(cp, &buf) catch {
                            try result.append(alloc, '?');
                            i += 6;
                            continue;
                        };
                        try result.appendSlice(alloc, buf[0..len]);
                        i += 6;
                    } else {
                        try result.append(alloc, '\\');
                        i += 1;
                    }
                },
                else => {
                    try result.append(alloc, '\\');
                    i += 1;
                },
            }
        } else {
            try result.append(alloc, raw[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(alloc);
}
