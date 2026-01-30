import cv2
import mediapipe as mp
import math
import time
import pygame
import random
from Pokemon import Pokemon


cap = cv2.VideoCapture(0)

screenHeight = 720
screenWidth = 1280

pokemOn = False
pokeballopened = False


cap.set(cv2.CAP_PROP_FRAME_WIDTH, screenWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screenHeight)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

current_time = time.time()

pygame.mixer.init()
backgroundMusic = pygame.mixer.Sound("music/background_music.mp3")
pop = pygame.mixer.Sound("music/ball-hit-pokemon.mp3")

sprite = cv2.imread("images/pokeball.png", cv2.IMREAD_UNCHANGED)
sprite = cv2.resize(sprite, (100, 100))
sh, sw = sprite.shape[:2]

grass = cv2.imread("images/grass.png", cv2.IMREAD_UNCHANGED)
grass = cv2.resize(grass, (300, 300))
gh, gw = grass.shape[:2]
gx1 = screenWidth - gw
gx2 = gx1 + gw
gy1 = screenHeight - gh
gy2 = gy1 + gh

pokemon_list = [
    Pokemon("Squirtle", "pokemon/squirtle.png", gx1, gy1),
    Pokemon("Bulbasaur", "pokemon/bulbasaur.png", gx1, gy1),
    Pokemon("Charmander", "pokemon/charmander.png", gx1, gy1),
    Pokemon("Pikachu", "pokemon/pikachu.png", gx1, gy1),
    Pokemon("Zapdos", "pokemon/zapdos.png", gx1, gy1),
    Pokemon("Articuno", "pokemon/articuno.png", gx1, gy1),
    Pokemon("Moltres", "pokemon/moltres.png", gx1, gy1),
    Pokemon("Raikou", "pokemon/raikou.png", gx1, gy1),
]

current_pokemon = random.choice(pokemon_list)

x1, y1 = 100, 100
x2, y2 = 200, 200
grabbing = False
prev_pinch = None
vx, vy = 0, 0
damping = .95
distp = 5.0


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

    alpha = sprite_crop[:, :, 3] / 255.0
    overlay = sprite_crop[:, :, :3]

    for c in range(3):
        roi[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]

    bg[y1:y2, x1:x2] = roi
    return bg


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                dx = thumb.x - index.x
                dy = thumb.y - index.y
                dist = math.sqrt(dx*dx + dy*dy)

                dpx = thumb.x - pinky.x
                dpy = thumb.y - pinky.y
                distp = math.sqrt(dpx*dpx + dpy*dpy)

                pinch_x = int((thumb.x + index.x) * 0.5 * frame.shape[1])
                pinch_y = int((thumb.y + index.y) * 0.5 * frame.shape[0])

                pinchp_x = int((thumb.x + pinky.x) * 0.5 * frame.shape[1])
                pinchp_y = int((thumb.y + pinky.y) * 0.5 * frame.shape[0])


                if dist < 0.1:
                    grabbing = True
                else:
                    grabbing = False

                if grabbing:
                    if prev_pinch is not None:
                        vx = pinch_x - prev_pinch[0]
                        vy = pinch_y - prev_pinch[1]
                    prev_pinch = (pinch_x, pinch_y)

                    if pinch_x > x1 - 20 and pinch_x < x2 + 20 and pinch_y > y1 - 20 and pinch_y < y2 + 20:
                        x1 = pinch_x - sw // 2
                        y1 = pinch_y - sh // 2
                        x2 = x1 + sw
                        y2 = y1 + sh
                else:
                    prev_pinch = None

        if not grabbing and x1 + vx >= 0 and x2 + vx <= screenWidth and y1 + vy >= 0 and y2 + vy <= screenHeight:
            x1 += int(vx)
            y1 += int(vy)
            x2 = x1 + sw
            y2 = y1 + sh
            vx *= damping
            vy *= damping

            if x1 >= current_pokemon.x1 - 50 and x2 <= current_pokemon.x1 + current_pokemon.w + 50 and \
                y1 >= current_pokemon.y1 - 50 and y2 <= current_pokemon.y1 + current_pokemon.h + 50 and pokemOn and pokeballopened is False:

                sprite = cv2.imread("images/open_pokeball.png", cv2.IMREAD_UNCHANGED)
                sprite = cv2.resize(sprite, (100, 100))
                pokeballopened = True
                pop.play()
                vx = 0
                vy = 0
                current_time = time.time()

            if time.time() - current_time > 2.0 and pokeballopened:
                current_time = time.time()
                pokeballopened = False
                sprite = cv2.imread("images/pokeball.png", cv2.IMREAD_UNCHANGED)
                sprite = cv2.resize(sprite, (100, 100))
                pokemOn = False
                current_pokemon = random.choice(pokemon_list)


        if backgroundMusic.get_num_channels() == 0:
            backgroundMusic.play()

        if distp < 0.1 and gx1 < pinchp_x < gx2 and gy1 < pinchp_y < gy2:
            pokemOn = True

        frame = draw_sprite(frame, grass, gx1, gy1)

        if pokemOn:
            current_pokemon.image[:, :, 3] = 255
            frame = draw_sprite(frame, current_pokemon.image, current_pokemon.x1, current_pokemon.y1)

        frame = draw_sprite(frame, sprite, x1, y1)
        cv2.imshow("Pokemon", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()