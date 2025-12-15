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

def inference_loop():
    global latest_result

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Run inference
        results = model.infer(frame, confidence=0.75)[0]

        detections = sv.Detections.from_inference(results)
        keypoints = sv.KeyPoints.from_inference(results)

        with result_lock:
            latest_result = (detections, keypoints)

        # Optional small sleep to avoid maxing CPU
        time.sleep(0.001)

# Start inference thread
thread = threading.Thread(target=inference_loop, daemon=True)
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update frame for inference
    with frame_lock:
        latest_frame = frame

    annotated = frame.copy()

    # Draw latest inference result (if available)
    with result_lock:
        if latest_result is not None:
            detections, keypoints = latest_result
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections)
            annotated = keypoint_annotator.annotate(
                annotated, key_points=keypoints
            )

    cv2.imshow("Live (Async Inference)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()