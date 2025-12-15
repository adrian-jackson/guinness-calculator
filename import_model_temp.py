import cv2
import supervision as sv
from inference import get_model

#Roboflow Settings
API_KEY = "YB4rtgGaV7id93lETrRc"
WORKSPACE = "guinnesscalculator"
PROJECT = "pose_calculator"
PROJECT_ID = "pose_calculator"
MODEL_VERSION = 7

# Load pre-trained YOLOv8n model from Roboflow
# Format: "workspace/project/version"
model_id = f"{WORKSPACE}/{PROJECT}/models/{MODEL_VERSION}"
model = get_model(model_id=PROJECT_ID, api_key=API_KEY)

# -------------------------
# Supervision annotators
# -------------------------
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# -------------------------
# Capture frames from webcam
# -------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on current frame
    results = model.infer(frame)[0]

    # Convert results to supervision Detections
    detections = sv.Detections.from_inference(results)

    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Display frame
    cv2.imshow("Roboflow Live Inference", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
