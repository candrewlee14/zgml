"""Per-op microbenchmark for MNIST CNN ops — PyTorch CPU (1 thread) baseline.

Matches the exact shapes from the zgml ConvClassifier:
  Conv2d(5x5, 1->8)  :  [32,1,28,28] * [8,1,5,5]  -> [32,8,24,24]
  ReLU                :  [32,8,24,24]               -> [32,8,24,24]
  MaxPool2d(2x2)      :  [32,8,24,24]               -> [32,8,12,12]
  FC (linear)         :  [32,1152] * [10,1152]^T    -> [32,10]
  CrossEntropyLoss    :  [32,10] + labels[32]        -> scalar

Run with: uv run --with torch benchmarks/op_bench_pytorch.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

BATCH = 32
WARMUP = 20
ITERS = 200


def bench(name, fn, *, shapes=""):
    """Benchmark a callable, returning (fwd_us, bwd_us)."""
    # Warmup
    for _ in range(WARMUP):
        fn()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)

    times.sort()
    min_us = times[0] / 1000
    p50_us = times[len(times) // 2] / 1000
    p90_us = times[len(times) * 9 // 10] / 1000
    mean_us = sum(times) / len(times) / 1000
    print(f"  {name:<22} {min_us:>10.1f} {p50_us:>10.1f} {p90_us:>10.1f} {mean_us:>10.1f}   {shapes}")
    return mean_us


def main():
    print(f"\nPer-Op Microbenchmark — PyTorch CPU 1-thread (batch={BATCH})")
    print("=" * 62)
    print()

    # ── Conv2d forward + backward ──
    print(f"  {'op':<22} {'min_us':>10} {'p50_us':>10} {'p90_us':>10} {'mean_us':>10}   shapes")
    print(f"  {'':-<22} {'':->10} {'':->10} {'':->10} {'':->10}")

    conv = nn.Conv2d(1, 8, 5, bias=False)
    x_conv = torch.randn(BATCH, 1, 28, 28, requires_grad=True)

    def conv_fwd():
        return conv(x_conv)

    bench("conv2d_fwd", conv_fwd, shapes="[32,1,28,28]*[8,1,5,5]->[32,8,24,24]")

    def conv_fwd_bwd():
        out = conv(x_conv)
        out.backward(torch.ones_like(out))
        if x_conv.grad is not None:
            x_conv.grad = None
        conv.weight.grad = None

    bench("conv2d_fwd+bwd", conv_fwd_bwd)

    # ── ReLU forward + backward ──
    x_relu = torch.randn(BATCH, 8, 24, 24, requires_grad=True)

    def relu_fwd():
        return F.relu(x_relu)

    bench("relu_fwd", relu_fwd, shapes="[32,8,24,24]->[32,8,24,24]")

    def relu_fwd_bwd():
        out = F.relu(x_relu)
        out.backward(torch.ones_like(out))
        x_relu.grad = None

    bench("relu_fwd+bwd", relu_fwd_bwd)

    # ── MaxPool2d forward + backward ──
    pool = nn.MaxPool2d(2)
    x_pool = torch.randn(BATCH, 8, 24, 24, requires_grad=True)

    def pool_fwd():
        return pool(x_pool)

    bench("maxpool_fwd", pool_fwd, shapes="[32,8,24,24]->[32,8,12,12]")

    def pool_fwd_bwd():
        out = pool(x_pool)
        out.backward(torch.ones_like(out))
        x_pool.grad = None

    bench("maxpool_fwd+bwd", pool_fwd_bwd)

    # ── FC (Linear) forward + backward ──
    fc = nn.Linear(1152, 10)
    x_fc = torch.randn(BATCH, 1152, requires_grad=True)

    def fc_fwd():
        return fc(x_fc)

    bench("fc_fwd", fc_fwd, shapes="[32,1152]*[10,1152]->[32,10]")

    def fc_fwd_bwd():
        out = fc(x_fc)
        out.backward(torch.ones_like(out))
        x_fc.grad = None
        fc.weight.grad = None
        fc.bias.grad = None

    bench("fc_fwd+bwd", fc_fwd_bwd)

    # ── CrossEntropyLoss forward + backward ──
    logits = torch.randn(BATCH, 10, requires_grad=True)
    labels = torch.randint(0, 10, (BATCH,))

    def xent_fwd():
        return F.cross_entropy(logits, labels)

    bench("xent_fwd", xent_fwd, shapes="[32,10]+labels->[1]")

    def xent_fwd_bwd():
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        logits.grad = None

    bench("xent_fwd+bwd", xent_fwd_bwd)

    # ── Full model forward + backward ──
    print()
    print("  Full model (Conv->ReLU->Pool->FC->XEnt):")
    print(f"  {'':-<22} {'':->10} {'':->10} {'':->10} {'':->10}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, 5)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(8 * 12 * 12, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    x_full = torch.randn(BATCH, 1, 28, 28)
    y_full = torch.randint(0, 10, (BATCH,))

    def full_fwd():
        return model(x_full)

    bench("model_fwd", full_fwd, shapes="[32,1,28,28]->[32,10]")

    def full_fwd_bwd():
        out = model(x_full)
        loss = criterion(out, y_full)
        loss.backward()
        for p in model.parameters():
            p.grad = None

    mean_total = bench("model_fwd+bwd", full_fwd_bwd)
    imgs_per_sec = BATCH / (mean_total / 1_000_000)
    print(f"\n  Throughput: {imgs_per_sec:.0f} img/s (at mean)")

    print(f"\n({WARMUP} warmup + {ITERS} timed iterations per op)\n")


if __name__ == "__main__":
    main()
