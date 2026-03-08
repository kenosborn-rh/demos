import json
import time
import numpy as np
import cv2
import onnxruntime as ort

MODEL_PATH = "out/placards.onnx"
LABELS_PATH = "out/labels.json"

CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720
IN_W, IN_H = 224, 224

# --- Safety / arming knobs ---
ALLOWED_COMMANDS = {"start", "stop", "slow", "reverse"}

ARM_THRESHOLD = 0.85
STABLE_FRAMES = 3
COOLDOWN_SEC = 1.0

NONE_MAX_PROB = 0.20
MARGIN_MIN = 0.20

# UI
SHOW_WINDOW = True
PRINT_EVERY_SEC = 0.20
DRAW_CROP_GUIDE = True
TRIGGER_FLASH_SEC = 0.75


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def center_square_crop(frame_bgr):
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]
    return crop, (x0, y0, side)


def preprocess_bgr(frame_bgr):
    crop, _ = center_square_crop(frame_bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def main():
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    if "none" not in labels:
        raise SystemExit("labels.json does not contain 'none' class.")
    none_idx = labels.index("none")

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try CAM_INDEX=1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    last_print = 0.0

    # stability / trigger state
    streak_label = None
    streak = 0
    last_trigger_time = 0.0
    last_trigger_label = None
    last_trigger_display_until = 0.0

    # fps estimate
    fps_t0 = time.time()
    fps_frames = 0
    fps_val = 0.0

    print("Running ARMED webcam inference. Press 'q' to quit.")
    print(
        f"ARM_THRESHOLD={ARM_THRESHOLD}  "
        f"STABLE_FRAMES={STABLE_FRAMES}  "
        f"COOLDOWN_SEC={COOLDOWN_SEC}"
    )
    print(f"NONE_MAX_PROB={NONE_MAX_PROB}  MARGIN_MIN={MARGIN_MIN}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        now = time.time()

        x = preprocess_bgr(frame)
        logits = sess.run(None, {input_name: x})[0][0]
        probs = softmax(logits)

        top2 = np.argsort(probs)[::-1][:2]
        i1, i2 = int(top2[0]), int(top2[1])
        label1, conf1 = labels[i1], float(probs[i1])
        label2, conf2 = labels[i2], float(probs[i2])

        none_prob = float(probs[none_idx])
        margin = conf1 - conf2

        frame_armed = (
            (label1 in ALLOWED_COMMANDS)
            and (conf1 >= ARM_THRESHOLD)
            and (none_prob <= NONE_MAX_PROB)
            and (margin >= MARGIN_MIN)
        )

        if frame_armed:
            if label1 == streak_label:
                streak += 1
            else:
                streak_label = label1
                streak = 1
        else:
            streak_label = None
            streak = 0

        triggered = False
        trigger_label = None

        if streak >= STABLE_FRAMES and (now - last_trigger_time) >= COOLDOWN_SEC:
            triggered = True
            trigger_label = streak_label
            last_trigger_time = now

            last_trigger_label = trigger_label
            last_trigger_display_until = now + TRIGGER_FLASH_SEC

            print(f"\n########## TRIGGER! {trigger_label.upper()} ##########\n", flush=True)

            streak_label = None
            streak = 0

        # FPS
        fps_frames += 1
        if fps_frames >= 15:
            dt = time.time() - fps_t0
            fps_val = fps_frames / max(dt, 1e-6)
            fps_t0 = time.time()
            fps_frames = 0

        # Logging
        if now - last_print >= PRINT_EVERY_SEC:
            status = "ARMED" if frame_armed else "idle "
            trig_txt = f"TRIGGER! {trigger_label}" if triggered else ""
            print(
                f"{label1:8s} conf={conf1:.3f}  "
                f"top2={label2}:{conf2:.3f}  "
                f"none={none_prob:.3f}  margin={margin:.3f}  "
                f"fps~{fps_val:4.1f}  {status}  streak={streak}  {trig_txt}"
            )
            last_print = now

        if SHOW_WINDOW:
            overlay = frame.copy()

            box_color = (255, 255, 255)  # white default
            if now < last_trigger_display_until:
                box_color = (0, 255, 0)      # green = triggered
            elif frame_armed:
                box_color = (0, 255, 255)    # yellow = armed

            if DRAW_CROP_GUIDE:
                _, (x0, y0, side) = center_square_crop(frame)
                cv2.rectangle(overlay, (x0, y0), (x0 + side, y0 + side), box_color, 3)

            text1 = f"{label1} {conf1:.3f} | top2 {label2} {conf2:.3f}"
            text2 = f"none {none_prob:.3f} | margin {margin:.3f} | streak {streak} | fps {fps_val:.1f}"

            if now < last_trigger_display_until and last_trigger_label is not None:
                text3 = f"TRIGGERED: {last_trigger_label.upper()}"
            elif frame_armed:
                text3 = "ARMED"
            else:
                text3 = "idle"

            cv2.putText(
                overlay, text1, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                overlay, text2, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                overlay, text3, (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 3, cv2.LINE_AA
            )

            cv2.imshow("Placard ARMED (q to quit)", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
