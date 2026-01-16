import cv2
from ultralytics import YOLO

# Load the YOLO model (small and fast)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

x1 = 100
y1 = 100
x2 = 200
y2 = 200


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw landmarks
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_x = int(right_hand.x * frame.shape[1])
        right_y = int(right_hand.y * frame.shape[0])
        # This is the place to implement the ball's picture
        left_x = int(left_hand.x * frame.shape[1])
        left_y = int(left_hand.y * frame.shape[0])

        if x1 < right_x < x1 + 10 and y1 < right_y < y2 or x1 < left_x < x1 + 10 and y1 < left_y < y2:
            x1 += 10
            x2 += 10
        if x2 - 10 < right_x < x2 and y1 < right_y < y2 or x2 - 10 < left_x < x2 and y1 < left_y < y2:
            x1 -= 10
            x2 -= 10
        if y1 < right_y < y1 + 10 and x1 < right_x < x2 or y1 < left_y < y1 + 10 and x1 < left_x < x2:
            y1 += 10
            y2 += 10
        if y2 - 10 < right_y < y2 and x1 < right_x < x2 or y2 - 10 < left_y < y2 and x1 < left_y < x2:
            y1 -= 10
            y2 -= 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking on the frame
    results = model.track(frame, persist=True)

    # Draw results on the frame
    annotated_frame = results[0].plot()


    # Show the frame
    cv2.imshow("YOLO Tracking Test", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# This test script captures video from the webcam, applies YOLOv8 tracking,