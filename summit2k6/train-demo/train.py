import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


@dataclass
class TrainConfig:
    data_dir: str
    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    seed: int
    out_pt: str
    out_labels: str
    use_mps: bool


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no cuda


def get_device(use_mps: bool) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path containing train/ and val/ folders")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-pt", default="out/placards.pt")
    ap.add_argument("--out-labels", default="out/labels.json")
    ap.add_argument("--use-mps", action="store_true", help="Use Apple Metal (MPS) if available")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        out_pt=args.out_pt,
        out_labels=args.out_labels,
        use_mps=args.use_mps,
    )

    set_seed(cfg.seed)
    device = get_device(cfg.use_mps)
    print("Device:", device)

    train_path = os.path.join(cfg.data_dir, "train")
    val_path = os.path.join(cfg.data_dir, "val")
    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise SystemExit(f"Expected {train_path} and {val_path} to exist")

    # Mild augmentations that match your “placard in front of webcam” reality.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        # NOTE: No mean/std normalization here to keep inference simpler.
        # If you later add normalization, you MUST match it in your MS-01 inference container.
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    classes = train_ds.classes
    num_classes = len(classes)
    print("Classes:", classes)
    os.makedirs(os.path.dirname(cfg.out_pt), exist_ok=True)

    # Save labels now (so you can carry them with the ONNX model later).
    with open(cfg.out_labels, "w") as f:
        json.dump(classes, f, indent=2)
    print("Wrote:", cfg.out_labels)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=False)

    # Model: MobileNetV3 Small (great for CPU inference)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze backbone initially (fast, reduces overfitting)
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=cfg.lr)

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in tqdm(train_dl, desc=f"train {epoch}/{cfg.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        val_acc = correct / max(total, 1)
        avg_loss = running_loss / max(len(train_dl), 1)
        print(f"epoch={epoch} train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                },
                cfg.out_pt
            )
            print(f"  ✅ saved best model to {cfg.out_pt} (val_acc={best_val_acc:.4f})")

    print("Done. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
