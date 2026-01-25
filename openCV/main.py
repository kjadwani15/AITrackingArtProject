import cv2
from ultralytics import YOLO
import mediapipe as mp
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

screenHeight = 720
screenWidth = 1280

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screenWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screenHeight)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

sprite = cv2.imread("pokeball.png", cv2.IMREAD_UNCHANGED)
sprite = cv2.resize(sprite, (100, 100))
sh, sw = sprite.shape[:2]

wasGrabbed = False
x1, y1 = 100, 100
x2, y2 = 200, 200
grabbing = False
prev_pinch = None
vx, vy = 0, 0
damping = 0.95


def draw_sprite(bg, sprite, x, y):
    h, w = sprite.shape[:2]
    frame_h, frame_w = bg.shape[:2]

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
        roi[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]

    bg[y1:y2, x1:x2] = roi
    return bg

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                dx = thumb.x - index.x
                dy = thumb.y - index.y
                dist = math.sqrt(dx*dx + dy*dy)

                pinch_x = int((thumb.x + index.x) * 0.5 * frame.shape[1])
                pinch_y = int((thumb.y + index.y) * 0.5 * frame.shape[0])

                if grabbing:
                    if prev_pinch is not None:
                        vx = pinch_x - prev_pinch[0]
                        vy = pinch_y - prev_pinch[1]
                    prev_pinch = (pinch_x, pinch_y)
                else:
                    prev_pinch = None

                if not grabbing:
                    if x1 + int(vx) > 0 and x2 + int(vx) < screenWidth:
                        x1 += int(vx)
                        x2 = x1 + sw
                    if y1 + int(vy) > 0 and y2 + int(vy) < screenHeight:
                        y1 += int(vy)
                        y2 = y1 + sh
                    vx = int(vx * damping)
                    vy = int(vy * damping)

                if dist < 0.05:   # pinching
                    grabbing = True
                else:
                    grabbing = False

                if grabbing:
                    if pinch_x > x1 and pinch_x < x2 and pinch_y > y1 and pinch_y < y2:
                        if pinch_x > 50 and pinch_x < screenWidth - 50 and pinch_y > 50 and pinch_y < screenHeight - 50:
                            x1 = pinch_x - sw // 2
                            y1 = pinch_y - sh // 2
                            x2 = x1 + sw
                            y2 = y1 + sh
        frame = draw_sprite(frame, sprite, x1, y1)
        cv2.imshow("Pose Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
