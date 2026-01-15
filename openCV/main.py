import cv2
from ultralytics import YOLO

# Load the YOLO model (small and fast)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking on the frame
    results = model.track(frame, persist=True)

    # Draw results on the frame
    annotated_frame = results[0].plot()
    cv2.rectangle(annotated_frame, (100, 100), (200, 200), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Tracking Test", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# This test script captures video from the webcam, applies YOLOv8 tracking,