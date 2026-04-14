"""MNIST CNN benchmark — PyTorch CPU baseline for comparison with zgml."""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

torch.set_num_threads(1)  # single-thread for fair comparison

# Same architecture as zgml: Conv(5x5, 1->8) -> ReLU -> MaxPool(2x2) -> FC(1152->10)
class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 5, padding=0)  # 28x28 -> 24x24
        self.pool = nn.MaxPool2d(2)                  # 24x24 -> 12x12
        self.fc = nn.Linear(8 * 12 * 12, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    print("\nMNIST CNN Training Benchmark (PyTorch CPU, 1 thread)")
    print("=" * 52, "\n")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("data_pytorch", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("data_pytorch", train=False, download=True, transform=transform)

    batch_size = 32
    n_epochs = 10

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_ds)} images, Test: {len(test_ds)} images")
    print(f"  Image size: 28x28\n")
    print("Architecture: Conv(5x5, 1->8) -> ReLU -> MaxPool(2x2) -> FC(1152->10)")
    print(f"Optimizer: Adam (lr=1e-3)")
    print(f"Batch size: {batch_size}, Epochs: {n_epochs}\n")

    model = ConvClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_start = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"  epoch {epoch+1}/{n_epochs} batch {batch_idx+1}/{len(train_loader)}"
                      f" loss={running_loss/(batch_idx+1):.4f}"
                      f" train_acc={100.*correct/total:.1f}%", end="\r")

        epoch_ms = (time.perf_counter() - epoch_start) * 1000
        avg_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        imgs_per_sec = total / (epoch_ms / 1000)

        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}"
              f"  train_acc={train_acc:.1f}%"
              f"  {epoch_ms:.0f}ms  ({imgs_per_sec:.0f} img/s)")

    total_ms = (time.perf_counter() - total_start) * 1000

    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100.0 * test_correct / test_total

    print(f"\nResults")
    print(f"-------")
    print(f"  Test accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
    print(f"  Total training time: {total_ms:.0f}ms")
    print(f"  Avg time per epoch: {total_ms/n_epochs:.0f}ms")
    print()

if __name__ == "__main__":
    main()
