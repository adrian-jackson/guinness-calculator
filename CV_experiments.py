from ultralytics import YOLO
import cv2
import math
import numpy

#idea: use YOLOv8 and fine-tune with ~50 images for each detection problem

#PROBLEM 1: CAN POUR ANGLE

#keypoint detection
can_kp_model = YOLO("yolov8n-pose.pt")   # pretrained pose model
glass_kp_model = YOLO("yolov8n-pose.pt")
glass_fill_model = YOLO("yolov8n-pose.pt")

# Size options. nano (n suffix) probably will work best bc of smaller train dataset size needed and faster inference times
# model = YOLO("yolov8n-pose.pt")     # tiny model (fastest)
# model = YOLO("yolov8s-pose.pt")   # small model
# model = YOLO("yolov8m-pose.pt")   # medium
# model = YOLO("yolov8l-pose.pt")   # large/accurate


def compute_angle_from_kp(kp1, kp2):
    """
    kp1, kp2: (x, y) coordinates of two keypoints
    Returns angle in degrees relative to horizontal axis.
    """
    x1, y1 = kp1
    x2, y2 = kp2
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

#PROBLEM 2: FOAM/LIQUID RATIO AND GLASS FILL
#need to detect three bounding boxes: liquid, foam, and empty space. compute the ratio of the three. 


#PROBLEM 3: GLASS HOLD ANGLE


#LIVE LOOP
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    can_results = can_kp_model(frame, verbose=False)

    annotated = frame.copy()

    if len(results) > 0:
        result = results[0]

        # YOLO keypoints shape: (num_objects, num_keypoints, 2)
        if result.keypoints is not None:
            kps = result.keypoints.xy

            for obj_kps in kps:
                # Expect exactly 2 keypoints: bottom & neck
                if obj_kps.shape[0] >= 2:
                    bottom = tuple(obj_kps[0])
                    neck = tuple(obj_kps[1])

                    # Draw keypoints
                    cv2.circle(annotated, bottom, 6, (0, 255, 0), -1)
                    cv2.circle(annotated, neck, 6, (0, 255, 0), -1)
                    cv2.line(annotated, bottom, neck, (0, 255, 0), 2)

                    # Compute tilt angle
                    angle = compute_angle(bottom, neck)

                    # Feedback
                    if angle < IDEAL_ANGLE - TOLERANCE:
                        feedback = "Tilt more ↑"
                        color = (0, 0, 255)
                    elif angle > IDEAL_ANGLE + TOLERANCE:
                        feedback = "Tilt less ↓"
                        color = (0, 0, 255)
                    else:
                        feedback = "Perfect ✔"
                        color = (0, 255, 0)

                    # Overlay text
                    cv2.putText(annotated, f"Angle: {angle:.1f} deg",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2)

                    cv2.putText(annotated, feedback,
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                color, 3)

    cv2.imshow("Bottle Tilt Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()