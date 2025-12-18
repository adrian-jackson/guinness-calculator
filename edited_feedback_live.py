# TO DO: Use pose bb to crop ROIs before segmentation for fill

from inference import get_model
import supervision as sv
import cv2
import numpy as np
import threading
import time

# -----------------------
# Video capture settings
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera (index 0). Check permissions or try index 1.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -----------------------
# Models
# -----------------------
pose_model = get_model(
    model_id="pose_calculator/11",
    api_key="YB4rtgGaV7id93lETrRc"
)

fill_model = get_model(
    model_id="fill_calculator/3",
    api_key="YB4rtgGaV7id93lETrRc"
)

# -----------------------
# Annotators (optional)
# -----------------------
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
keypoint_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.GREEN)

# -----------------------
# Timing / pacing
# -----------------------
target_fps = 2
period = 1.0 / target_fps
next_t = time.perf_counter()

# -----------------------
# Shared state
# -----------------------
latest_frame = None
latest_pose_result = None          # (detections, keypoints, angles)
latest_fill_result = None          # dict: det_index -> metrics dict

frame_lock = threading.Lock()
pose_lock = threading.Lock()
fill_lock = threading.Lock()

stop_event = threading.Event()

POSE_CLASSES = {0: "can", 1: "glass"}
FILL_CLASSES = {0: "empty", 1: "foam", 2: "liquid"}

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def pick_best_index(detections, class_name: str):
    """Pick the highest-confidence detection index for a given class."""
    idxs = [i for i, cid in enumerate(detections.class_id) if POSE_CLASSES.get(int(cid)) == class_name]
    if not idxs:
        return None
    # confidence might not exist for every backend; be defensive
    try:
        conf = detections.confidence
        best = max(idxs, key=lambda i: float(conf[i]))
        return best
    except Exception:
        # fallback: first occurrence
        return idxs[0]

def fill_inference_loop():
    global latest_fill_result

    while not stop_event.is_set():
        # get latest frame
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        # need both frame + pose detections to crop ROIs
        with pose_lock:
            pose_snapshot = latest_pose_result

        if frame is None or pose_snapshot is None:
            time.sleep(0.01)
            continue

        detections, _, _ = pose_snapshot

        h, w = frame.shape[:2]
        fill_metrics = {}

        for i in range(len(detections)):
            if POSE_CLASSES.get(int(detections.class_id[i])) != "glass":
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            box = clamp_box(x1, y1, x2, y2, w, h)
            if box is None:
                continue

            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            try:
                seg_results = fill_model.infer(roi, confidence=0.75)[0]
                seg_det = sv.Detections.from_inference(seg_results)
            except Exception as e:
                # don't let the thread die
                print(f"[fill] infer failed: {e}")
                continue

            foam_pixels = 0
            liquid_pixels = 0
            empty_pixels = 0

            # sum mask areas per class (mask is boolean array)
            for j in range(len(seg_det)):
                class_name = FILL_CLASSES.get(int(seg_det.class_id[j]))
                mask = seg_det.mask[j]
                if mask is None:
                    continue
                area = int(mask.sum())

                if class_name == "foam":
                    foam_pixels += area
                elif class_name == "liquid":
                    liquid_pixels += area
                elif class_name == "empty":
                    empty_pixels += area

            total = foam_pixels + liquid_pixels + empty_pixels
            if total <= 0:
                continue

            foam_frac = foam_pixels / total
            liquid_frac = liquid_pixels / total
            foam_to_liquid = foam_pixels / max(liquid_pixels, 1)  # avoid div0

            fill_metrics[i] = {
                "foam_frac": foam_frac,
                "liquid_frac": liquid_frac,
                "foam_to_liquid": foam_to_liquid,
                "foam_pixels": foam_pixels,
                "liquid_pixels": liquid_pixels,
                "total_pixels": total,
                "ts": time.time(),
            }

            # Debug (optional)
            # print(f"[fill] det={i} total={total} liquid={liquid_pixels} foam={foam_pixels} foam/liquid={foam_to_liquid:.3f}")

        with fill_lock:
            latest_fill_result = fill_metrics

        # segmentation intentionally slow
        time.sleep(2.0)

def pose_inference_loop():
    global latest_pose_result

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        try:
            results = pose_model.infer(frame, confidence=0.75)[0]
            detections = sv.Detections.from_inference(results)
            keypoints = sv.KeyPoints.from_inference(results)
        except Exception as e:
            print(f"[pose] infer failed: {e}")
            time.sleep(0.05)
            continue

        angles = []
        for i in range(len(detections)):
            class_name = POSE_CLASSES.get(int(detections.class_id[i]))
            if class_name not in ("can", "glass"):
                angles.append(None)
                continue

            # keypoints.xy shape: (num_dets, K, 2)
            try:
                kp = keypoints.xy[i]
                if kp is None or kp.shape[0] < 2:
                    angles.append(None)
                    continue

                x1, y1 = kp[0]
                x2, y2 = kp[1]
                angle = float(np.degrees(np.atan2(y2 - y1, x2 - x1)))
                angles.append(angle)
            except Exception:
                angles.append(None)

        with pose_lock:
            latest_pose_result = (detections, keypoints, angles)

        time.sleep(0.001)

# -----------------------
# Start threads
# -----------------------
pose_thread = threading.Thread(target=pose_inference_loop, daemon=True)
fill_thread = threading.Thread(target=fill_inference_loop, daemon=True)
pose_thread.start()
fill_thread.start()

try:
    while True:
        # pace capture loop
        now = time.perf_counter()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += period

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break

        with frame_lock:
            latest_frame = frame

        annotated = frame.copy()

        # snapshot results without holding locks too long
        with pose_lock:
            pose_snapshot = latest_pose_result
        with fill_lock:
            fill_snapshot = latest_fill_result or {}

        feedback = None

        if pose_snapshot is not None:
            detections, keypoints, angles = pose_snapshot

            # Optional debug annotations
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections)
            annotated = keypoint_annotator.annotate(annotated, key_points=keypoints)

            can_idx = pick_best_index(detections, "can")
            glass_idx = pick_best_index(detections, "glass")

            # Only compute feedback if we have everything we need
            if can_idx is not None and glass_idx is not None:

                can_angle = angles[can_idx] if can_idx < len(angles) else None
                glass_angle = angles[glass_idx] if glass_idx < len(angles) else None

                metrics = fill_snapshot.get(glass_idx)

                if metrics != None: 
                    cv2.putText(
                        annotated,
                        "Glass is empty...",
                        (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                if can_angle is not None and glass_angle is not None and metrics is not None:
                    # Use foam/liquid ratio or foam fraction depending on what you want
                    ratio = metrics["foam_to_liquid"]
                    foam_frac = metrics["foam_frac"]

                    adjusted_angle = abs(can_angle) - (90 - abs(glass_angle))

                    # --- feedback rules (tune thresholds) ---
                    # Example: "too much liquid" if you have low foam fraction
                    if 135 <= adjusted_angle <= 180 and foam_frac < 0.20:
                        feedback = "Too much liquid — tilt MORE"
                    elif 90 <= adjusted_angle <= 135 and foam_frac > 0.25:
                        feedback = "Too much foam — tilt LESS"
                    else:
                        feedback = "Good pour 👍"

                    x1, y1, _, _ = detections.xyxy[can_idx]
                    cv2.putText(
                        annotated,
                        feedback,
                        (int(x1), int(y1) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255) if "Too much" in feedback else (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
        
        # If we can't compute feedback yet, still keep UI alive
        if feedback is None:
            cv2.putText(
                annotated,
                "Waiting for can+glass+fill...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Live (Async Inference)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_event.set()
    pose_thread.join(timeout=1.0)
    fill_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
