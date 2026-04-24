import json
import os
import time
import threading
import numpy as np
import cv2
import onnxruntime as ort
from flask import Flask, Response

# ---------------------------------------------------------------------------
# Configuration — all values read from environment variables so the container
# can be tuned at runtime via the Quadlet unit without rebuilding the image.
# Defaults here match the original MacBook development values.
# ---------------------------------------------------------------------------

def _bool(key: str, default: bool) -> bool:
    """Parse a boolean env var accepting true/false/1/0 (case-insensitive)."""
    val = os.environ.get(key, "").strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default

MODEL_PATH  = os.environ.get("MODEL_PATH",  "/app/models/placards.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "/app/models/labels.json")

CAM_INDEX   = int(os.environ.get("CAMERA_INDEX", "0"))
CAM_W       = int(os.environ.get("CAM_W", "640"))
CAM_H       = int(os.environ.get("CAM_H", "480"))
EXPOSURE    = int(os.environ.get("EXPOSURE", "-1"))  # -1 = auto, positive = manual
IN_W, IN_H  = 224, 224  # model input size — fixed by training, not tunable

DISPLAY_SIZE = int(os.environ.get("DISPLAY_SIZE", "800"))
WEB_PORT     = int(os.environ.get("WEB_PORT", "8080"))

# --- Safety / arming knobs ---
ALLOWED_COMMANDS = {"start", "stop", "slow", "reverse"}

ARM_THRESHOLD = float(os.environ.get("ARM_THRESHOLD", "0.85"))
STABLE_FRAMES = int(os.environ.get("STABLE_FRAMES",  "3"))
COOLDOWN_SEC  = float(os.environ.get("COOLDOWN_SEC", "1.0"))

NONE_MAX_PROB = float(os.environ.get("NONE_MAX_PROB", "0.20"))
MARGIN_MIN    = float(os.environ.get("MARGIN_MIN",    "0.20"))

# --- Bright paper gate ---
USE_PAPER_GATE    = _bool("USE_PAPER_GATE", True)
MIN_BRIGHT_FRAC   = float(os.environ.get("MIN_BRIGHT_FRAC",   "0.18"))
MAX_BRIGHT_FRAC   = float(os.environ.get("MAX_BRIGHT_FRAC",   "0.92"))
MIN_CENTER_BRIGHT = float(os.environ.get("MIN_CENTER_BRIGHT", "0.12"))
BRIGHT_THRESH     = int(os.environ.get("BRIGHT_THRESH",       "185"))

PRINT_EVERY_SEC      = float(os.environ.get("PRINT_EVERY_SEC",   "0.20"))
DRAW_CROP_GUIDE      = _bool("DRAW_CROP_GUIDE",     True)
TRIGGER_FLASH_SEC    = float(os.environ.get("TRIGGER_FLASH_SEC", "0.90"))
SHOW_DEBUG_ON_SCREEN = _bool("SHOW_DEBUG_ON_SCREEN", True)

# --- MQTT ---
MQTT_ENABLED  = _bool("MQTT_ENABLED", False)
MQTT_BROKER   = os.environ.get("MQTT_BROKER",  "localhost")
MQTT_PORT     = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC    = os.environ.get("MQTT_TOPIC",   "train/cmd")
DEVICE_ID     = os.environ.get("DEVICE_ID",    "ms01-camera")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "placards-v1")

# ---------------------------------------------------------------------------
# Shared frame buffer — inference loop writes, Flask thread reads
# ---------------------------------------------------------------------------
_frame_lock   = threading.Lock()
_latest_frame = None  # JPEG bytes


def set_frame(jpeg_bytes):
    global _latest_frame
    with _frame_lock:
        _latest_frame = jpeg_bytes


def get_frame():
    with _frame_lock:
        return _latest_frame


# ---------------------------------------------------------------------------
# Flask MJPEG server
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <title>AI Inference</title>
  <style>
    body { margin: 0; background: #000; display: flex;
           justify-content: center; align-items: flex-start; }
    img  { display: block; max-width: 100vw; max-height: 100vh; }
  </style>
</head>
<body>
  <img src="/stream" />
</body>
</html>"""


@app.route("/stream")
def stream():
    def generate():
        while True:
            frame = get_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)  # ~30fps max to client
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# Inference helpers — identical to inference.py
# ---------------------------------------------------------------------------

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


def preprocess_rgb(rgb):
    rgb = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def detect_bright_paper(square_bgr):
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    bright_frac = float(np.count_nonzero(mask)) / mask.size
    h, w = mask.shape[:2]
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = mask[y0:y1, x0:x1]
    center_bright_frac = float(np.count_nonzero(center)) / center.size
    found = (
        bright_frac >= MIN_BRIGHT_FRAC and
        bright_frac <= MAX_BRIGHT_FRAC and
        center_bright_frac >= MIN_CENTER_BRIGHT
    )
    return found, mask, bright_frac, center_bright_frac


def command_label_text(label: str) -> str:
    return label.upper()


def draw_text_with_bg(img, text, org, font_scale=1.0, text_color=(255, 255, 255),
                      bg_color=(0, 0, 0), thickness=2, pad=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(
        img,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        bg_color,
        -1,
    )
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def inference_loop():
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    if "none" not in labels:
        raise SystemExit("labels.json does not contain 'none' class.")
    none_idx = labels.index("none")

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try CAM_INDEX=1.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if EXPOSURE > 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
        print(f"Camera exposure: manual ({EXPOSURE})")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto mode
        print("Camera exposure: auto")

    last_print = 0.0
    streak_label = None
    streak = 0
    last_trigger_time = 0.0
    last_trigger_label = None
    last_trigger_display_until = 0.0

    fps_t0 = time.time()
    fps_frames = 0
    fps_val = 0.0

    status_h = 300

    gate_status = "paper gate ON" if USE_PAPER_GATE else "paper gate OFF"
    print(f"Running ARMED webcam inference [{gate_status}]. Web stream on :{WEB_PORT}")
    print(
        f"ARM_THRESHOLD={ARM_THRESHOLD}  STABLE_FRAMES={STABLE_FRAMES}  "
        f"COOLDOWN_SEC={COOLDOWN_SEC}"
    )
    print(
        f"USE_PAPER_GATE={USE_PAPER_GATE}  BRIGHT_THRESH={BRIGHT_THRESH}  "
        f"MIN_BRIGHT_FRAC={MIN_BRIGHT_FRAC}  MIN_CENTER_BRIGHT={MIN_CENTER_BRIGHT}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        now = time.time()

        square, (gx, gy, gside) = center_square_crop(frame)

        paper_found = True
        paper_mask = None
        bright_frac = 0.0
        center_bright_frac = 0.0

        if USE_PAPER_GATE:
            paper_found, paper_mask, bright_frac, center_bright_frac = \
                detect_bright_paper(square)

        if paper_found:
            rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            x = preprocess_rgb(rgb)
            logits = sess.run(None, {input_name: x})[0][0]
            probs = softmax(logits)
        else:
            probs = np.zeros(len(labels), dtype=np.float32)
            probs[none_idx] = 1.0

        top2 = np.argsort(probs)[::-1][:2]
        i1, i2 = int(top2[0]), int(top2[1])
        label1, conf1 = labels[i1], float(probs[i1])
        label2, conf2 = labels[i2], float(probs[i2])

        none_prob = float(probs[none_idx])
        margin = conf1 - conf2

        frame_armed = (
            paper_found and
            (label1 in ALLOWED_COMMANDS) and
            (conf1 >= ARM_THRESHOLD) and
            (none_prob <= NONE_MAX_PROB) and
            (margin >= MARGIN_MIN)
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

            if MQTT_ENABLED:
                try:
                    import paho.mqtt.publish as mqtt_publish
                    payload = json.dumps({
                        "cmd":        trigger_label,
                        "confidence": round(float(conf1), 4),
                        "device":     DEVICE_ID,
                        "model":      MODEL_VERSION,
                    })
                    mqtt_publish.single(
                        MQTT_TOPIC,
                        payload=payload,
                        hostname=MQTT_BROKER,
                        port=MQTT_PORT,
                    )
                    print(f"  MQTT → {MQTT_BROKER}:{MQTT_PORT} {MQTT_TOPIC} {payload}", flush=True)
                except Exception as mqtt_err:
                    print(f"  MQTT ERROR: {mqtt_err}", flush=True)

            streak_label = None
            streak = 0

        # FPS
        fps_frames += 1
        if fps_frames >= 15:
            dt = time.time() - fps_t0
            fps_val = fps_frames / max(dt, 1e-6)
            fps_t0 = time.time()
            fps_frames = 0

        # Terminal logging
        if now - last_print >= PRINT_EVERY_SEC:
            status = "ARMED" if frame_armed else "idle "
            trig_txt = f"TRIGGER! {trigger_label}" if triggered else ""
            print(
                f"{label1:8s} conf={conf1:.3f}  "
                f"top2={label2}:{conf2:.3f}  "
                f"none={none_prob:.3f}  margin={margin:.3f}  "
                f"paper={'yes' if paper_found else 'no '}  "
                f"bright={bright_frac:.3f} center={center_bright_frac:.3f}  "
                f"fps~{fps_val:4.1f}  {status}  streak={streak}  {trig_txt}"
            )
            last_print = now

        # ── Build canvas — identical layout to inference.py ──────────────────
        h_frame, w_frame = frame.shape[:2]
        feed_size    = min(h_frame, w_frame)
        canvas_w     = DISPLAY_SIZE
        canvas_h     = DISPLAY_SIZE + status_h
        canvas       = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Center-crop frame to square then scale to DISPLAY_SIZE
        x0_crop     = max(0, (w_frame - feed_size) // 2)
        y0_crop     = max(0, (h_frame - feed_size) // 2)
        feed_w      = min(feed_size, w_frame)
        feed_h      = min(feed_size, h_frame)
        feed_square = frame[y0_crop:y0_crop + feed_h, x0_crop:x0_crop + feed_w]
        feed_scaled = cv2.resize(feed_square, (DISPLAY_SIZE, DISPLAY_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        canvas[:DISPLAY_SIZE, :] = feed_scaled

        # Guide box coords scaled to DISPLAY_SIZE
        scale      = DISPLAY_SIZE / feed_size
        gbx        = int((gx - x0_crop) * scale)
        gby        = int((gy - y0_crop) * scale)
        gside_disp = int(gside * scale)

        # Paper gate mask preview
        if USE_PAPER_GATE and paper_mask is not None:
            preview_size = 120
            mask_small = cv2.resize(paper_mask, (preview_size, preview_size),
                                    interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            px = 8
            py = DISPLAY_SIZE - preview_size - 8
            canvas[py:py + preview_size, px:px + preview_size] = mask_small
            cv2.rectangle(canvas, (px, py), (px + preview_size, py + preview_size),
                          (255, 255, 255), 1)
            draw_text_with_bg(canvas, "paper gate", (px + 4, py + 16),
                font_scale=0.45, text_color=(255, 255, 255),
                bg_color=(0, 0, 0), thickness=1, pad=3)

        # Guide box color
        box_color = (255, 255, 255)
        if now < last_trigger_display_until:
            box_color = (0, 255, 0)
        elif frame_armed:
            box_color = (0, 255, 255)

        if DRAW_CROP_GUIDE:
            cv2.rectangle(canvas, (gbx, gby), (gbx + gside_disp, gby + gside_disp), box_color, 3)

        if USE_PAPER_GATE:
            inner_color = (0, 180, 0) if paper_found else (0, 0, 180)
            inset = 12
            cv2.rectangle(
                canvas,
                (gbx + inset, gby + inset),
                (gbx + gside_disp - inset, gby + gside_disp - inset),
                inner_color, 2
            )

        # Status panel
        panel_y = DISPLAY_SIZE
        cv2.line(canvas, (0, panel_y), (canvas_w, panel_y), (60, 60, 60), 2)

        if now < last_trigger_display_until and last_trigger_label is not None:
            status_text  = "COMMAND SENT:"
            cmd_text     = command_label_text(last_trigger_label)
            status_color = (0, 255, 0)
        elif frame_armed:
            status_text  = "READY:"
            cmd_text     = command_label_text(label1)
            status_color = (0, 255, 255)
        else:
            status_text  = "PREDICTION:"
            cmd_text     = command_label_text(label1)
            status_color = (255, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for txt, fs, thick, yoff, color in [
            (status_text,               0.9, 2,  50,  status_color),
            (cmd_text,                  2.0, 4,  130, status_color),
            (f"Confidence: {conf1:.0%}", 0.9, 2, 185, (200, 200, 200)),
        ]:
            (tw, _), _ = cv2.getTextSize(txt, font, fs, thick)
            cx = (canvas_w - tw) // 2
            draw_text_with_bg(canvas, txt, (cx, panel_y + yoff),
                font_scale=fs, text_color=color, bg_color=(0, 0, 0),
                thickness=thick, pad=6)

        if SHOW_DEBUG_ON_SCREEN:
            debug_lines = [
                f"top2: {label2} {conf2:.3f}   |   none: {none_prob:.3f}   |   margin: {margin:.3f}",
                f"streak: {streak}   |   paper: {'off' if not USE_PAPER_GATE else ('yes' if paper_found else 'no')}   |   bright: {bright_frac:.3f}/{center_bright_frac:.3f}   |   fps: {fps_val:.1f}",
            ]
            for i, line in enumerate(debug_lines):
                (tw, _), _ = cv2.getTextSize(line, font, 0.52, 1)
                cx = (canvas_w - tw) // 2
                draw_text_with_bg(canvas, line, (cx, panel_y + 220 + i * 32),
                    font_scale=0.52, text_color=(160, 160, 160),
                    bg_color=(0, 0, 0), thickness=1, pad=3)

        # Encode canvas as JPEG and push to shared frame buffer
        _, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        set_frame(jpeg.tobytes())

    cap.release()


# ---------------------------------------------------------------------------
# Entry point — start inference in background thread, Flask in foreground
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)

