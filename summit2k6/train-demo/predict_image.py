import argparse
import json
import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    x = np.array(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    x = np.expand_dims(x, axis=0)   # add batch dim

    return x


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="out/placards.onnx")
    parser.add_argument("--labels", default="out/labels.json")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    # Load labels
    with open(args.labels) as f:
        labels = json.load(f)

    # Load ONNX model
    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Prepare image
    x = preprocess(args.image)

    # Run inference
    outputs = session.run(None, {input_name: x})
    logits = outputs[0][0]

    probs = softmax(logits)
    idx = np.argmax(probs)

    print()
    print("Prediction:", labels[idx])
    print("Confidence:", float(probs[idx]))
    print()

    # show top 3
    print("Top predictions:")
    top = np.argsort(probs)[::-1][:3]
    for i in top:
        print(f"{labels[i]:10s} {probs[i]:.4f}")


if __name__ == "__main__":
    main()
