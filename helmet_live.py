
from datetime import datetime
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

try:
    import serial
except ImportError:
    serial = None

BASE_DIR = Path(__file__).resolve().parent

HELMET_PATH = BASE_DIR / "runs/detect/train5/weights/best.pt"
TRIPLE_PATH = BASE_DIR / "runs/detect/triple_riding_fresh/weights/best.pt"
PLATE_PATH = BASE_DIR / "runs/detect/number_plate_fresh/weights/best.pt"

HELMET_EVIDENCE = BASE_DIR / "evidence/helmet"
TRIPLE_EVIDENCE = BASE_DIR / "evidence/triple_riding_clean"
PLATE_EVIDENCE = BASE_DIR / "evidence/number_plate"

for folder in (HELMET_EVIDENCE, TRIPLE_EVIDENCE, PLATE_EVIDENCE):
    folder.mkdir(parents=True, exist_ok=True)

CAM_WIDTH, CAM_HEIGHT = 1280, 720
SAVE_COOLDOWN = 5

# ESP32 serial settings. Change only if your Mac shows a different port.
SERIAL_PORT = "/dev/cu.usbserial-A5069RR4"
SERIAL_BAUD = 115200
CLEAR_DELAY = 2.0  # must be violation-free this long before gate reopens


def load_model(path, name):
    if not path.exists():
        print(f"[WARNING] {name} model missing: {path}")
        return None
    print(f"[LOAD] {name}: {path}")
    return YOLO(str(path))


def no_helmet(label):
    label = label.lower().replace("_", " ").replace("-", " ")
    return "no helmet" in label or "without helmet" in label or "nohelmet" in label


def detect(model, frame, kind):
    if model is None:
        return []

    # Keep the two main detectors lighter.
    # Use higher resolution for number plates only when needed.
    if kind == "helmet":
        conf, imgsz, classes = 0.20, 416, None
    elif kind == "triple":
        conf, imgsz, classes = 0.35, 416, [2]
    else:
        conf, imgsz, classes = 0.10, 640, [2]

    results = model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        classes=classes,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    result = results[0]
    names = result.names
    frame_h, frame_w = frame.shape[:2]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        if isinstance(names, dict):
            raw_label = str(names.get(cls_id, f"class_{cls_id}"))
        else:
            raw_label = str(names[cls_id])

        if kind == "triple":
            if cls_id != 2:
                continue
            label, color = "Triple Riding", (0, 165, 255)
        elif kind == "plate":
            if cls_id != 2:
                continue

            label, color = "Number Plate", (255, 100, 220)

        else:
            label = raw_label

            if no_helmet(label):
                color = (0, 0, 255)
            elif "helmet" in label.lower():
                color = (0, 255, 0)
            else:
                color = (255, 190, 0)

        detections.append(
            (x1, y1, x2, y2, label, score, color)
        )

    return detections

  
 


def draw_detections(frame, groups):
    output = frame.copy()

    for detections in groups.values():
        for x1, y1, x2, y2, label, score, color in detections:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"{label} {score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    return output


def save_evidence(folder, prefix, frame):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = folder / f"{prefix}_{stamp}.jpg"

    if cv2.imwrite(str(path), frame):
        print(f"[EVIDENCE SAVED] {path}")
        return True

    print(f"[WARNING] Could not save {path}")
    return False


def main():
    helmet_model = load_model(HELMET_PATH, "Helmet")
    triple_model = load_model(TRIPLE_PATH, "Triple riding")
    plate_model = load_model(PLATE_PATH, "Number plate")

    if not any((helmet_model, triple_model, plate_model)):
        print("[ERROR] No models could be loaded.")
        return

    # Connect to ESP32. Camera detection still runs if serial is unavailable.
    esp = None
    if serial is None:
        print("[WARNING] pyserial missing. Install with: python3 -m pip install pyserial")
    else:
        try:
            esp = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2, write_timeout=1)
            time.sleep(2)  # opening serial may reset the ESP32
            esp.reset_input_buffer()
            esp.write(b"CLEAR\n")
            esp.flush()
            print(f"[ESP32] Connected: {SERIAL_PORT}")
            print("[ESP32] Sent CLEAR (green LED on, gate open)")
        except Exception as exc:
            print(f"[WARNING] ESP32 serial connection failed: {exc}")
            print("[WARNING] Check SERIAL_PORT, FTDI connection, and close Arduino Serial Monitor.")
            esp = None

    violation_sent = False
    last_violation_time = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[READY] Standalone camera running.")
    print("Press Q or ESC to quit.")
    print("Helmet evidence:", HELMET_EVIDENCE)
    print("Triple evidence:", TRIPLE_EVIDENCE)
    print("Number-plate evidence:", PLATE_EVIDENCE)

    last_saved = {
        "helmet": 0.0,
        "triple": 0.0,
        "plate": 0.0,
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Camera frame unavailable.")
                break

            # All three models analyse THIS SAME FRAME, keeping boxes aligned.
            
            # First run the two lighter models on the same frame.
            detections = {
                "helmet": detect(helmet_model, frame, "helmet"),
                "triple": detect(triple_model, frame, "triple"),
                "plate": [],
            }

            no_helmet_found = any(
                no_helmet(d[4]) for d in detections["helmet"]
            )
            triple_found = bool(detections["triple"])

            # Only run the more expensive plate model when a
            # helmet or triple-riding violation is detected.
            violation_found = no_helmet_found or triple_found

            if violation_found:
                detections["plate"] = detect(
                    plate_model, frame, "plate"
                )

            plate_found = bool(detections["plate"])


            violation_found = no_helmet_found or triple_found

            # Hardware response: red LED + close gate on either violation.
            # Reopen only after the scene is clear for CLEAR_DELAY seconds.
            now_for_hardware = time.time()
            if violation_found:
                last_violation_time = now_for_hardware
                if esp is not None and not violation_sent:
                    try:
                        esp.write(b"VIOLATION\n")
                        esp.flush()
                        print("[ESP32] VIOLATION sent: red LED on, green off, gate closing")
                        violation_sent = True
                    except Exception as exc:
                        print(f"[WARNING] ESP32 command failed: {exc}")
            elif violation_sent and now_for_hardware - last_violation_time >= CLEAR_DELAY:
                if esp is not None:
                    try:
                        esp.write(b"CLEAR\n")
                        esp.flush()
                        print("[ESP32] CLEAR sent: red LED off, green on, gate opening")
                    except Exception as exc:
                        print(f"[WARNING] ESP32 command failed: {exc}")
                violation_sent = False

            if triple_found:
                status, status_color = "TRIPLE RIDING CANDIDATE", (0, 165, 255)
            elif no_helmet_found:
                status, status_color = "NO HELMET DETECTED", (0, 0, 255)
            else:
                status, status_color = "MONITORING", (0, 220, 120)

            annotated = draw_detections(frame, detections)

            cv2.rectangle(
                annotated, (0, 0), (annotated.shape[1], 32), (8, 16, 28), -1
            )
            cv2.putText(
                annotated,
                f"NEXUS | {status} | Q: quit",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                2,
                cv2.LINE_AA,
            )

            now = time.time()

            # Separate helmet-violation evidence.
            if no_helmet_found and now - last_saved["helmet"] >= SAVE_COOLDOWN:
                if save_evidence(HELMET_EVIDENCE, "no_helmet", annotated):
                    last_saved["helmet"] = now

            # Separate triple-riding evidence.
            if triple_found and now - last_saved["triple"] >= SAVE_COOLDOWN:
                if save_evidence(TRIPLE_EVIDENCE, "triple_riding", annotated):
                    last_saved["triple"] = now

            # Save plate evidence only when a violation and a plausible plate
            # are detected together in the same frame.
            if (
                violation_found
                and plate_found
                and now - last_saved["plate"] >= SAVE_COOLDOWN
            ):
                if save_evidence(PLATE_EVIDENCE, "number_plate", annotated):
                    last_saved["plate"] = now

            cv2.imshow("NEXUS Road Safety - Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if esp is not None:
            try:
                esp.write(b"CLEAR\n")
                esp.flush()
                esp.close()
            except Exception:
                pass
        print("[STOPPED] Camera released.")


if __name__ == "__main__":
    main()
