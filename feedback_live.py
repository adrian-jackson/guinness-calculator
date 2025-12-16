#TO DO: Use pose bb to crop ROIs before segmentation for fill:

from inference import get_model
import supervision as sv
import cv2
import numpy as np
import threading
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pose_model = get_model(
    model_id="pose_calculator/14",
    api_key="YB4rtgGaV7id93lETrRc"
)

fill_model = get_model(
    model_id="fill_calculator/3",
    api_key="YB4rtgGaV7id93lETrRc"
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
keypoint_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.GREEN)

target_fps = 2
period = 1/target_fps
next_t = time.perf_counter()

latest_frame = None
latest_pose_result = None
latest_fill_result = None
running = True

frame_lock = threading.Lock()
pose_lock = threading.Lock()
fill_lock = threading.Lock()

POSE_CLASSES = {0: "can", 1: "glass"}
FILL_CLASSES = {0: "empty", 1: "foam", 2: "liquid"}

def fill_inference_loop():
    global latest_fill_result

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        with pose_lock:
            if latest_pose_result is None:
                continue
            detections, _, _ = latest_pose_result

        fill_ratios = {}

        for i in range(len(detections)):
            if POSE_CLASSES.get(detections.class_id[i]) != "glass":
                continue

            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            seg_results = fill_model.infer(roi, confidence=0.75)[0]
            seg_det = sv.Detections.from_inference(seg_results)

            total_pixels = 0
            liquid_pixels = 0
            foam_pixels = 0

            for j in range(len(seg_det)):
                class_name = FILL_CLASSES.get(seg_det.class_id[j])
                mask = seg_det.mask[j]

                area = int(mask.sum())

                if class_name == "foam":
                    foam_pixels += area

                if class_name == "liquid":
                    liquid_pixels += area

            if liquid_pixels > 0:
                fill_ratios[i] =  foam_pixels / liquid_pixels
            print('total pixels', total_pixels, '\nliquid pixels', liquid_pixels, '\nfoam pixels', foam_pixels)
        with fill_lock:
            latest_fill_result = fill_ratios

        time.sleep(2)  # segmentation is intentionally slow

def pose_inference_loop():
    global latest_pose_result

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = pose_model.infer(frame, confidence=0.6)[0]
        detections = sv.Detections.from_inference(results)
        keypoints = sv.KeyPoints.from_inference(results)

        angles = []

        for i in range(len(detections)):
            class_name = POSE_CLASSES.get(detections.class_id[i])

            if class_name not in ("can", "glass"):
                angles.append(None)
                continue

            kp = keypoints.xy[i]
            if kp.shape[0] < 2:
                angles.append(None)
                continue

            x1, y1 = kp[0]
            x2, y2 = kp[1]

            angle = np.degrees(np.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        with pose_lock:
            latest_pose_result = (detections, keypoints, angles)

        time.sleep(0.001)

# Start inference thread
pose_thread = threading.Thread(target=pose_inference_loop, daemon=True)
fill_thread = threading.Thread(target=fill_inference_loop, daemon=True)
pose_thread.start()
fill_thread.start()

while True:

    now = time.perf_counter()
    if now < next_t:
        time.sleep(next_t - now)
    next_t += period

    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        latest_frame = frame

    annotated = frame.copy()

    with pose_lock:
        if latest_pose_result is not None:
            detections, keypoints, angles = latest_pose_result

            #enable for annotations
            '''annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections)
            annotated = keypoint_annotator.annotate(
                annotated, key_points=keypoints
            )'''

            with fill_lock:
                fill_ratios = latest_fill_result or {}

            # --- associate can & glass ---
            can_idx = None
            glass_idx = None

            for i, cls_id in enumerate(detections.class_id):
                if POSE_CLASSES.get(cls_id) == "can":
                    can_idx = i
                elif POSE_CLASSES.get(cls_id) == "glass":
                    glass_idx = i

            if can_idx is None or glass_idx is None:
                pass
            else:
                can_angle = angles[can_idx]
                glass_angle = angles[glass_idx]

            ratio = fill_ratios.get(glass_idx)
            if ratio is None:
                continue

            # --- adjusted angle ---
            adjusted_angle = abs(can_angle) - (90 - abs(glass_angle))

            # --- feedback rules ---
            if 135 <= adjusted_angle <= 180 :#and ratio >= 0.22:
                feedback = "Too much liquid — tilt MORE"

            elif 90 <= adjusted_angle <= 135: # and ratio < 0.25:
                feedback = "Too much foam — tilt LESS"

            else:
                feedback = "Good pour 👍"

            # --- draw feedback ---
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

            # enable for annotations
            '''text = f"{POSE_CLASSES[detections.class_id[i]]}: {angle:.1f}°"

            if i in fill_ratios:
                text += f" | Fill: {fill_ratios[i]*100:.0f}%"

            cv2.putText(
                annotated,
                text,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )'''

    cv2.imshow("Live (Async Inference)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
pose_thread.join(timeout=1.0)
fill_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()