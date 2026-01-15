import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose
from mediapipe.python.solutions import drawing_utils


pose_detector = pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)

    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            pose.POSE_CONNECTIONS
        )

    cv2.imshow("MediaPipe Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
