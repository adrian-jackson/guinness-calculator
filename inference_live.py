from inference import get_model
import supervision as sv
import cv2
import numpy as np
import threading
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

model = get_model(
    model_id="pose_calculator/7",
    api_key="YB4rtgGaV7id93lETrRc"
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
keypoint_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.GREEN)

latest_frame = None
latest_result = None
running = True

frame_lock = threading.Lock()
result_lock = threading.Lock()

CLASS_NAMES = {
    0: "can",
    1: "glass"
}

def inference_loop():
    global latest_result

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = model.infer(frame, confidence=0.75)[0]

        detections = sv.Detections.from_inference(results)
        keypoints = sv.KeyPoints.from_inference(results)

        angles = []  # one angle per detection

        for i in range(len(detections)):
            class_id = detections.class_id[i]
            class_name = CLASS_NAMES.get(class_id)

            # Only compute for can & glass
            if class_name not in ("can", "glass"):
                angles.append(None)
                continue

            kp = keypoints.xy[i]

            # Safety check
            if kp.shape[0] < 2:
                angles.append(None)
                continue

            x1, y1 = kp[0]
            x2, y2 = kp[1]

            dx = x2 - x1
            dy = y2 - y1

            angle_deg = np.degrees(np.atan2(dy, dx))
            angles.append(angle_deg)

        with result_lock:
            latest_result = (detections, keypoints, angles)

        time.sleep(0.001)

# Start inference thread
thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        latest_frame = frame

    annotated = frame.copy()

    with result_lock:
        if latest_result is not None:
            detections, keypoints, angles = latest_result

            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections)
            annotated = keypoint_annotator.annotate(
                annotated, key_points=keypoints
            )

            # 👇 draw angle text per object
            for i, angle in enumerate(angles):
                if angle is None:
                    continue

                x1, y1, x2, y2 = detections.xyxy[i]
                class_name = CLASS_NAMES.get(detections.class_id[i], "obj")

                text = f"{class_name}: {angle:.1f}°"

                cv2.putText(
                    annotated,
                    text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    cv2.imshow("Live (Async Inference)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()