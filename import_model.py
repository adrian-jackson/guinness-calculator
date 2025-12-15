from roboflow import Roboflow
import cv2
from PIL import Image
import os
import tempfile
import numpy as np

rf = Roboflow(api_key="YB4rtgGaV7id93lETrRc")

workspace = rf.workspace("guinnesscalculator")
project = workspace.project("pose_calculator")
model = project.version(5).model

cap = cv2.VideoCapture(0)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
            cv2.imwrite(path, frame)

        # Make prediction
        prediction = model.predict(path, confidence=40).json()
        os.remove(path)

        '''keypoints = prediction[0]['keypoints']
        if len(keypoints) >= 2:
            # Extract coordinates
            x1, y1 = keypoints[0]['x'], keypoints[0]['y']
            x2, y2 = keypoints[1]['x'], keypoints[1]['y']

            # Compute angle in degrees
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            print(angle)
            # Draw keypoints
            cv2.circle(frame, (int(x1), int(y1)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(x2), int(y2)), 5, (0, 0, 255), -1)

            # Draw line between keypoints
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Overlay angle text
            cv2.putText(
                frame,
                f"Angle: {angle:.2f} deg",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )'''

    frame_count += 1

    # Show the frame continuously
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()