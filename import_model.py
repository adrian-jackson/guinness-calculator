from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="YB4rtgGaV7id93lETrRc")

workspace = rf.workspace("guinnesscalculator")
project = workspace.project("pose_calculator")
model = project.version(5).model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show live video
    cv2.imshow("Webcam", frame)

    # Save frame
    if frame_count % 5 == 0:
        prediction = model.predict(frame, confidence=40, overlap=30).json()
        print(prediction)
    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()