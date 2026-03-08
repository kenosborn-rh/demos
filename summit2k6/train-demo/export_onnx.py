import argparse
import json
import os

import torch
from torch import nn
from torchvision import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt produced by train.py")
    ap.add_argument("--out", default="out/placards.onnx", help="Output ONNX path")
    ap.add_argument("--labels", default="out/labels.json", help="Path to labels.json (or will write from checkpoint)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    classes = ckpt.get("classes")
    if not classes:
        raise SystemExit("Checkpoint missing 'classes'. Re-run training or update checkpoint.")
    num_classes = len(classes)

    # Recreate same architecture
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    # Write labels
    os.makedirs(os.path.dirname(args.labels), exist_ok=True)
    with open(args.labels, "w") as f:
        json.dump(classes, f, indent=2)

    print("✅ Wrote ONNX:", args.out)
    print("✅ Wrote labels:", args.labels)
    print("Classes:", classes)


if __name__ == "__main__":
    main()
