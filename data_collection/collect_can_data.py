import cv2
import os
from datetime import datetime


cap = cv2.VideoCapture(0)

#edit this for different datatypes
SAVE_DIR = os.path.join("angle_data",f"{datetime.now().strftime("%Y%m%d_%H%M%S")}")
os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live video
    cv2.imshow("Webcam", frame)

    # Save frame
    if frame_count % 5 == 0:
        filename = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()