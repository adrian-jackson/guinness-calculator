from inference import get_model
import supervision as sv
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = get_model(
    model_id="pose_calculator/7",
    api_key="YB4rtgGaV7id93lETrRc"
)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
keypoint_annotator = sv.VertexAnnotator(
    radius=5,
    color=sv.Color.RED
)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 == 0:
        results = model.infer(frame)[0]

        # Bounding boxes
        detections = sv.Detections.from_inference(results)

        # Extract keypoints
        keypoints = sv.KeyPoints.from_inference(results)

        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections)
        #annotated = keypoint_annotator.annotate(annotated, key_points=keypoints)
        for kp in keypoints.xy:
            for x, y in kp:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.imshow("Live", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1