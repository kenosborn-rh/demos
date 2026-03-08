import argparse
import os
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--val-ratio", type=float, default=0.15)
args = parser.parse_args()

dataset = Path(args.dataset)
train_dir = dataset / "train"
val_dir = dataset / "val"

# remove existing val split
if val_dir.exists():
    shutil.rmtree(val_dir)

val_dir.mkdir(parents=True, exist_ok=True)

classes = [d.name for d in train_dir.iterdir() if d.is_dir()]

print("Classes:", classes)

for cls in classes:
    src = train_dir / cls
    dst = val_dir / cls
    dst.mkdir(parents=True, exist_ok=True)

    images = list(src.glob("*"))
    random.shuffle(images)

    val_count = int(len(images) * args.val_ratio)

    moved = 0
    for img in images[:val_count]:
        shutil.move(str(img), dst / img.name)
        moved += 1

    print(f"{cls}: total={len(images)} moved_to_val={moved}")

print("Validation split rebuilt.")
