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

sprite = cv2.imread("ball.png", cv2.IMREAD_UNCHANGED)  # must be RGBA
sprite = cv2.resize(sprite, (100, 100))
sh, sw = sprite.shape[:2]

cap = cv2.VideoCapture(0)

x1 = 100
y1 = 100
x2 = 200
y2 = 200

def draw_sprite(bg, sprite, x, y):
    h, w = sprite.shape[:2]
    frame_h, frame_w = bg.shape[:2]

    # Clip region to screen
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, frame_w)
    y2 = min(y + h, frame_h)

    sprite_x1 = x1 - x
    sprite_y1 = y1 - y
    sprite_x2 = sprite_x1 + (x2 - x1)
    sprite_y2 = sprite_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg

    roi = bg[y1:y2, x1:x2]
    sprite_crop = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]

    alpha = sprite_crop[:, :, 2] / 255.0
    overlay = sprite_crop[:, :, :3]

    for c in range(3):
        roi[:, :, c] = (alpha * overlay[:, :, c] +
                        (1 - alpha) * roi[:, :, c])

    bg[y1:y2, x1:x2] = roi
    return bg



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


        frame = draw_sprite(frame, sprite, x1, y1)

        cv2.imshow("Pose Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()