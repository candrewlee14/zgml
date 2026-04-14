//! Persistent inference for GPT models.
//!
//! `InferenceSession` is the main entry point.  It builds the computation
//! graph once, then re-executes it on every token with zero per-step
//! allocation — no graph rebuild, no weight copies, no KV-cache memcpy.
//!
//! ```
//! const Session = zgml.inference.InferenceSession(f32, config);
//!
//! var session = try Session.init(allocator);
//! defer session.deinit();
//!
//! // Load weights (writes directly into the session's model tensors).
//! checkpoint.load(f32, &session.model.params(), path);
//!
//! // Generate tokens.
//! for (0..max_tokens) |_| {
//!     const logits = try session.step(next_token);
//!     next_token = argmax(logits);
//! }
//!
//! // Start a new sequence.
//! session.reset();
//! ```
//!
//! Internally the session owns:
//!   - model weights (persistent arena)
//!   - KV caches (persistent arena, mutated in-place)
//!   - an `InferencePlan` (frozen `ComputeGraph` with workspace reuse)
//!
//! The plan patches four bound inputs each step:
//!   1. token embedding   (`[d_model, 1]`)
//!   2. positional encoding (`[d_model, 1]`)
//!   3. causal mask        (`[max_seq_len, 1]`, 0 / -inf)
//!   4. KV-cache write positions (storage_offset on slice_assign nodes)
//!
//! Then it calls `reset()` + `computeNoGrad()` on the frozen graph.

const std = @import("std");

const Tensor = @import("tensor.zig").Tensor;
const ComputeGraph = @import("graph.zig").ComputeGraph;
const GPT = @import("models/gpt.zig").GPT;
const GPTConfig = @import("models/gpt.zig").GPTConfig;

/// Frozen forward-only execution plan.
///
/// Built once from a `ComputeGraph` trace of `GPT.forwardCachedFrozen`.
/// Re-executed each step by patching bound inputs (token embedding,
/// positional encoding, causal mask, KV-cache write positions) and
/// calling `computeNoGrad` — no graph rebuild, no fusion re-analysis,
/// no per-step allocation.
///
/// Workspace liveness analysis assigns shared buffer slots to
/// intermediate tensors so dead buffers are recycled automatically.
pub fn InferencePlan(comptime T: type, comptime config: GPTConfig) type {
    const Model = GPT(T, config);
    const d_model = config.d_model;
    const max_seq = config.max_seq_len;

    return struct {
        const Self = @This();

        graph: ComputeGraph(T),
        backing_alloc: std.mem.Allocator,

        // Bound inputs — data patched before each execution.
        token_input: *Tensor(T),
        pos_input: *Tensor(T),
        attn_mask: *Tensor(T),

        // Cached slice_assign nodes — avoids linear scan per step.
        slice_assign_nodes: []const *Tensor(T),

        // Output tensor (read after execution).
        logits: *Tensor(T),
        logits_buf: []T,

        // Workspace reuse state.
        workspace_bufs: [][]T,

        /// Build a frozen plan from an existing model and KV caches.
        ///
        /// Traces one forward pass through `forwardCachedFrozen`, builds the
        /// graph, runs fusion, and optionally optimises the workspace.  The
        /// model weights and KV caches must outlive the plan (they are
        /// referenced, not copied).
        pub fn init(
            model: *const Model,
            k_caches: [config.n_layers][config.n_heads]*Tensor(T),
            v_caches: [config.n_layers][config.n_heads]*Tensor(T),
            backing_alloc: std.mem.Allocator,
        ) !Self {
            var graph = ComputeGraph(T).init(backing_alloc);
            errdefer graph.deinit();
            const a = graph.allocator();

            // Bound-input placeholders (leaves: op=.none, never zeroed by reset).
            const token_input = try Tensor(T).init(a, &.{ d_model, 1 });
            const pos_input = try Tensor(T).init(a, &.{ d_model, 1 });
            const attn_mask = try Tensor(T).init(a, &.{ max_seq, 1 });
            @memset(attn_mask.data, -std.math.inf(T));

            const x = token_input.add(pos_input);
            const logits = model.forwardCachedFrozen(x, k_caches, v_caches, 0, attn_mask);

            // Build forward graph + fusion (one-time cost).
            try graph.infer(logits);

            // Cache slice_assign node pointers for fast per-step patching.
            var sa_count: usize = 0;
            for (graph.nodes.items) |node| {
                if (node.opTag() == .slice_assign) sa_count += 1;
            }
            const sa_nodes = try backing_alloc.alloc(*Tensor(T), sa_count);
            errdefer backing_alloc.free(sa_nodes);
            var sa_idx: usize = 0;
            for (graph.nodes.items) |node| {
                if (node.opTag() == .slice_assign) {
                    sa_nodes[sa_idx] = node;
                    sa_idx += 1;
                }
            }

            // Stable output buffer — survives across steps.
            const logits_buf = try backing_alloc.alloc(T, config.vocab_size);
            errdefer backing_alloc.free(logits_buf);

            // Workspace reuse: share buffers among temporaries whose
            // lifetimes don't overlap.
            var bufs: [][]T = &.{};
            optimizeWorkspace(&graph, backing_alloc, &bufs) catch {};

            return .{
                .graph = graph,
                .backing_alloc = backing_alloc,
                .token_input = token_input,
                .pos_input = pos_input,
                .attn_mask = attn_mask,
                .slice_assign_nodes = sa_nodes,
                .logits = logits,
                .logits_buf = logits_buf,
                .workspace_bufs = bufs,
            };
        }

        /// Free the graph, workspace buffers, and output buffer.
        /// Does NOT free the model weights or KV caches (caller owns those).
        pub fn deinit(self: *Self) void {
            for (self.workspace_bufs) |buf| self.backing_alloc.free(buf);
            if (self.workspace_bufs.len > 0) self.backing_alloc.free(self.workspace_bufs);
            self.backing_alloc.free(self.logits_buf);
            self.backing_alloc.free(self.slice_assign_nodes);
            self.graph.deinit();
        }

        /// Execute one step: patch inputs, reset intermediates, compute.
        /// Returns a stable logits buffer that remains valid until the next step.
        pub fn execute(
            self: *Self,
            model: *const Model,
            token_id: usize,
            pos: usize,
        ) []const T {
            // 1. Patch token embedding.
            const tok_data = model.embed.token_embed.inner.data;
            @memcpy(
                self.token_input.data[0..d_model],
                tok_data[token_id * d_model ..][0..d_model],
            );

            // 2. Patch positional encoding.
            const pe_data = if (config.learnable_pos_embed)
                model.embed.pos_encode.inner.data
            else
                model.embed.pos_encode.data;
            @memcpy(
                self.pos_input.data[0..d_model],
                pe_data[pos * d_model ..][0..d_model],
            );

            // 3. Update causal mask: 0 for positions ≤ pos, -inf otherwise.
            for (self.attn_mask.data[0..max_seq], 0..) |*v, i| {
                v.* = if (i <= pos) 0 else -std.math.inf(T);
            }

            // 4. Patch KV-cache write positions.
            for (self.slice_assign_nodes) |node| node.storage_offset = pos;

            // 5. Reset intermediates and execute.
            self.graph.reset();
            self.graph.computeNoGrad();

            // 6. Copy to stable output buffer.
            @memcpy(self.logits_buf, self.logits.data[0..config.vocab_size]);
            return self.logits_buf;
        }

        // ---------------------------------------------------------------
        // Workspace optimisation
        // ---------------------------------------------------------------

        /// Analyse tensor liveness across the forward schedule and assign
        /// shared workspace buffers so that dead temporaries are recycled.
        fn optimizeWorkspace(
            graph: *ComputeGraph(T),
            alloc: std.mem.Allocator,
            out_bufs: *[][]T,
        ) !void {
            const nodes = graph.nodes.items[0..graph.forward_node_count];
            if (nodes.len == 0) return;

            // Build pointer → index map.
            var ptr_to_idx = std.AutoHashMap(*Tensor(T), u32).init(alloc);
            defer ptr_to_idx.deinit();
            try ptr_to_idx.ensureTotalCapacity(@intCast(nodes.len));
            for (nodes, 0..) |node, i| ptr_to_idx.putAssumeCapacity(node, @intCast(i));

            // Compute last-use step for each node.
            const last_use = try alloc.alloc(u32, nodes.len);
            defer alloc.free(last_use);
            for (0..nodes.len) |i| last_use[i] = @intCast(i);

            for (nodes, 0..) |node, step| {
                inline for (.{ node.src0, node.src1 }) |maybe_src| {
                    if (maybe_src) |s| {
                        if (ptr_to_idx.get(s)) |idx| {
                            last_use[idx] = @max(last_use[idx], @as(u32, @intCast(step)));
                        }
                    }
                }
            }

            // Identify workspace-eligible nodes: intermediate, owned data,
            // not fused-skip.
            const eligible = try alloc.alloc(bool, nodes.len);
            defer alloc.free(eligible);
            for (nodes, 0..) |node, i| {
                eligible[i] = node.opTag() != .none and
                    node.ownsData() and
                    !(i < graph.fused_skip.items.len and graph.fused_skip.items[i]);
            }

            // Slot assignment: best-fit (smallest adequate free slot).
            const Slot = struct { free_after: u32, size: usize };
            var slots: std.ArrayList(Slot) = .{};
            defer slots.deinit(alloc);

            const assignment = try alloc.alloc(i32, nodes.len);
            defer alloc.free(assignment);
            @memset(assignment, -1);

            for (nodes, 0..) |node, step_usize| {
                if (!eligible[step_usize]) continue;
                const step: u32 = @intCast(step_usize);
                const size = node.nElems();

                var best: ?usize = null;
                for (slots.items, 0..) |slot, si| {
                    if (slot.free_after < step and slot.size >= size) {
                        if (best == null or slot.size < slots.items[best.?].size) {
                            best = si;
                        }
                    }
                }

                if (best) |si| {
                    slots.items[si].free_after = last_use[step_usize];
                    assignment[step_usize] = @intCast(si);
                } else {
                    assignment[step_usize] = @intCast(slots.items.len);
                    try slots.append(alloc, .{ .free_after = last_use[step_usize], .size = size });
                }
            }

            if (slots.items.len == 0) return;

            // Allocate workspace buffers and redirect tensor data pointers.
            const bufs = try alloc.alloc([]T, slots.items.len);
            errdefer {
                for (bufs) |b| if (b.len > 0) alloc.free(b);
                alloc.free(bufs);
            }
            for (slots.items, 0..) |slot, si| {
                bufs[si] = try alloc.alloc(T, slot.size);
            }

            for (nodes, 0..) |node, i| {
                if (assignment[i] < 0) continue;
                const si: usize = @intCast(assignment[i]);
                node.data = bufs[si][0..node.nElems()];
            }

            out_bufs.* = bufs;
        }
    };
}

/// Persistent inference session backed by a frozen `InferencePlan`.
///
/// Owns model weights, KV caches, and the plan.  Each `step` call
/// patches bound inputs and re-executes — no graph rebuild, no
/// per-token allocation.
///
/// ```
/// const Session = InferenceSession(f32, my_config);
/// var s = try Session.init(allocator);
/// defer s.deinit();
///
/// // Optionally load pretrained weights into s.model
///
/// const logits = try s.step(token_id);   // [vocab_size]
/// s.reset();                              // clear KV caches, rewind to pos 0
/// ```
pub fn InferenceSession(comptime T: type, comptime config: GPTConfig) type {
    const Model = GPT(T, config);
    const Plan = InferencePlan(T, config);
    const d_head = config.d_model / config.n_heads;

    return struct {
        const Self = @This();

        backing_alloc: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        model: Model,
        k_caches: [config.n_layers][config.n_heads]*Tensor(T),
        v_caches: [config.n_layers][config.n_heads]*Tensor(T),
        plan: Plan,
        pos: usize,

        /// Create a session with freshly initialised weights.
        ///
        /// To use pretrained weights, overwrite them after init:
        /// ```
        /// var s = try Session.init(alloc);
        /// checkpoint.load(f32, &s.model.params(), path);
        /// ```
        pub fn init(backing_alloc: std.mem.Allocator) !Self {
            var arena = std.heap.ArenaAllocator.init(backing_alloc);
            errdefer arena.deinit();
            const a = arena.allocator();

            const model = try Model.init(a);

            var k_caches: [config.n_layers][config.n_heads]*Tensor(T) = undefined;
            var v_caches: [config.n_layers][config.n_heads]*Tensor(T) = undefined;
            for (0..config.n_layers) |l| {
                for (0..config.n_heads) |h| {
                    k_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    v_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    @memset(k_caches[l][h].data, 0);
                    @memset(v_caches[l][h].data, 0);
                }
            }

            var plan = try Plan.init(&model, k_caches, v_caches, backing_alloc);
            errdefer plan.deinit();

            return .{
                .backing_alloc = backing_alloc,
                .arena = arena,
                .model = model,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .plan = plan,
                .pos = 0,
            };
        }

        /// Free the plan, model arena, and all owned memory.
        pub fn deinit(self: *Self) void {
            self.plan.deinit();
            self.arena.deinit();
        }

        /// Clear KV caches and rewind to position 0.
        /// Call this to start generating a new sequence.
        pub fn reset(self: *Self) void {
            self.pos = 0;
            for (0..config.n_layers) |l| {
                for (0..config.n_heads) |h| {
                    @memset(self.k_caches[l][h].data, 0);
                    @memset(self.v_caches[l][h].data, 0);
                }
            }
        }

        /// Current sequence position (number of tokens processed so far).
        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        /// Process one token and return logits over the vocabulary.
        ///
        /// Returns a `[vocab_size]` slice that remains valid until the next
        /// call to `step`.  Returns `error.SequenceTooLong` if the KV cache
        /// is full (`pos >= max_seq_len`).
        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;
            const logits = self.plan.execute(&self.model, token_id, self.pos);
            self.pos += 1;
            return logits;
        }
    };
}

/// Eager per-step inference session (Phase 1 baseline).
///
/// Creates and destroys a `ComputeGraph` on every token.  Simpler but
/// slower than `InferenceSession` — kept as a reference implementation
/// and fallback.  Prefer `InferenceSession` for production use.
pub fn GPTDecodeSession(comptime T: type, comptime config: GPTConfig) type {
    const Model = GPT(T, config);
    const d_head = config.d_model / config.n_heads;

    return struct {
        const Self = @This();

        backing_alloc: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,
        model: Model,
        k_caches: [config.n_layers][config.n_heads]*Tensor(T),
        v_caches: [config.n_layers][config.n_heads]*Tensor(T),
        logits_buf: []T,
        pos: usize,

        pub fn init(backing_alloc: std.mem.Allocator) !Self {
            var arena = std.heap.ArenaAllocator.init(backing_alloc);
            errdefer arena.deinit();
            const a = arena.allocator();

            const model = try Model.init(a);

            var k_caches: [config.n_layers][config.n_heads]*Tensor(T) = undefined;
            var v_caches: [config.n_layers][config.n_heads]*Tensor(T) = undefined;
            for (0..config.n_layers) |l| {
                for (0..config.n_heads) |h| {
                    k_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    v_caches[l][h] = try Tensor(T).init(a, &.{ d_head, config.max_seq_len });
                    @memset(k_caches[l][h].data, 0);
                    @memset(v_caches[l][h].data, 0);
                }
            }

            const logits_buf = try a.alloc(T, config.vocab_size);

            return .{
                .backing_alloc = backing_alloc,
                .arena = arena,
                .model = model,
                .k_caches = k_caches,
                .v_caches = v_caches,
                .logits_buf = logits_buf,
                .pos = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
        }

        pub fn reset(self: *Self) void {
            self.pos = 0;
            for (0..config.n_layers) |l| {
                for (0..config.n_heads) |h| {
                    @memset(self.k_caches[l][h].data, 0);
                    @memset(self.v_caches[l][h].data, 0);
                }
            }
        }

        pub fn position(self: *const Self) usize {
            return self.pos;
        }

        pub fn step(self: *Self, token_id: usize) ![]const T {
            if (self.pos >= config.max_seq_len) return error.SequenceTooLong;

            var g = ComputeGraph(T).init(self.backing_alloc);
            defer g.deinit();

            const logits = self.model.forwardCached(g.allocator(), token_id, self.pos, self.k_caches, self.v_caches);
            try g.infer(logits);
            @memcpy(self.logits_buf, logits.data[0..self.logits_buf.len]);
            self.pos += 1;
            return self.logits_buf;
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "InferenceSession produces valid logits" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const tokens = [_]usize{ 0, 3, 1 };
    for (tokens) |tok| {
        const logits = try session.step(tok);
        try testing.expectEqual(@as(usize, cfg.vocab_size), logits.len);
        for (logits) |v| {
            try testing.expect(!std.math.isNan(v));
            try testing.expect(!std.math.isInf(v));
        }
    }
}

test "InferenceSession matches manual forward" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };
    const Model = GPT(f32, cfg);
    const d_head = cfg.d_model / cfg.n_heads;

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    // Build a reference model with identical weights in a separate arena.
    var ref_arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer ref_arena.deinit();
    const ra = ref_arena.allocator();
    const ref_model = try Model.init(ra);
    for (session.model.params(), ref_model.params()) |src, dst| @memcpy(dst.data, src.data);

    var ref_k: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    var ref_v: [cfg.n_layers][cfg.n_heads]*Tensor(f32) = undefined;
    for (0..cfg.n_layers) |l| for (0..cfg.n_heads) |h| {
        ref_k[l][h] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len });
        ref_v[l][h] = try Tensor(f32).init(ra, &.{ d_head, cfg.max_seq_len });
        @memset(ref_k[l][h].data, 0);
        @memset(ref_v[l][h].data, 0);
    };

    // Build a reference attn mask in the ref arena.
    const ref_mask = try Tensor(f32).init(ra, &.{ cfg.max_seq_len, 1 });

    const tokens = [_]usize{ 2, 5, 0, 7, 1 };
    for (tokens, 0..) |tok, pos| {
        // Reference: manual frozen forward in a temporary graph.
        var g = ComputeGraph(f32).init(testing.allocator);
        defer g.deinit();
        const ga = g.allocator();

        const tok_input = try Tensor(f32).init(ga, &.{ cfg.d_model, 1 });
        const pos_input = try Tensor(f32).init(ga, &.{ cfg.d_model, 1 });
        const tok_data = ref_model.embed.token_embed.inner.data;
        @memcpy(tok_input.data[0..cfg.d_model], tok_data[tok * cfg.d_model ..][0..cfg.d_model]);
        const pe_data = ref_model.embed.pos_encode.data;
        @memcpy(pos_input.data[0..cfg.d_model], pe_data[pos * cfg.d_model ..][0..cfg.d_model]);
        for (ref_mask.data[0..cfg.max_seq_len], 0..) |*v, i| {
            v.* = if (i <= pos) 0 else -std.math.inf(f32);
        }

        const x = tok_input.add(pos_input);
        const ref_logits = ref_model.forwardCachedFrozen(x, ref_k, ref_v, pos, ref_mask);
        try g.infer(ref_logits);

        // Frozen session step.
        const session_logits = try session.step(tok);
        for (session_logits, ref_logits.data[0..cfg.vocab_size]) |got, want| {
            try testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

test "InferenceSession reset replays identical outputs" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 1,
        .max_seq_len = 8,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const tokens = [_]usize{ 1, 2, 3 };
    var first: [tokens.len][cfg.vocab_size]f32 = undefined;

    for (tokens, 0..) |tok, i| {
        const logits = try session.step(tok);
        @memcpy(&first[i], logits);
    }

    session.reset();

    for (tokens, 0..) |tok, i| {
        const logits = try session.step(tok);
        for (logits, first[i][0..]) |got, want| {
            try testing.expectApproxEqAbs(want, got, 1e-6);
        }
    }
}

test "InferencePlan workspace reuse reduces slot count" {
    const cfg = GPTConfig{
        .vocab_size = 8,
        .d_model = 4,
        .n_heads = 2,
        .d_ff = 8,
        .n_layers = 2,
        .max_seq_len = 16,
    };

    var session = try InferenceSession(f32, cfg).init(testing.allocator);
    defer session.deinit();

    const n_slots = session.plan.workspace_bufs.len;
    const n_forward = session.plan.graph.forward_node_count;

    var n_intermediates: usize = 0;
    for (session.plan.graph.nodes.items[0..n_forward]) |node| {
        if (node.opTag() != .none and node.ownsData()) n_intermediates += 1;
    }

    if (n_intermediates > 2) {
        try testing.expect(n_slots < n_intermediates);
    }
}
