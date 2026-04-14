const std = @import("std");
const testing = std.testing;
pub const Linear = @import("models/linear.zig").Model;
pub const Poly = @import("models/poly.zig").Model;
pub const TransformerBlock = @import("models/transformer.zig").TransformerBlock;
pub const Embedding = @import("models/embedding.zig").Embedding;
pub const GPT = @import("models/gpt.zig").GPT;
pub const GPTConfig = @import("models/gpt.zig").GPTConfig;
pub const gpt_loader = @import("models/gpt_loader.zig");
pub const XorMlp = @import("models/xor_mlp.zig").Model;
pub const MlpClassifier = @import("models/mlp_classifier.zig").Model;
pub const Autoencoder = @import("models/autoencoder.zig").Model;
pub const ConvClassifier = @import("models/conv_classifier.zig").Model;

test "ref all decls" {
    _ = testing.refAllDecls(Linear(f32));
    _ = testing.refAllDecls(Poly(f32));
    _ = @import("models/transformer.zig");
    _ = @import("models/embedding.zig");
    _ = @import("models/gpt.zig");
    _ = testing.refAllDecls(XorMlp(f32));
    _ = testing.refAllDecls(MlpClassifier(f32));
    _ = testing.refAllDecls(Autoencoder(f32));
    _ = testing.refAllDecls(ConvClassifier(f32));
}
