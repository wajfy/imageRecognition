from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
from pynput.keyboard import Controller
import time

WINDOW_SCALE = 0.7
CONFIDENCE_THRESHOLD = 0.2
IMGSZ = 1024
RESET_DELAY = 1.0  # seconds
SHOW_WINDOW = True

keyboard = Controller()

model = YOLO("runs/detect/train/weights/best.pt")
model.to("cuda")

BoxSizeW, BoxSizeH = 40, 90
REGIONS = {
    "A": (750, 800, BoxSizeW, BoxSizeH),     # left
    "W": (900, 645, BoxSizeH, BoxSizeW),     # up
    "D": (1132, 800, BoxSizeW, BoxSizeH)     # right
}
KEY_MAP = {"A": "a", "W": "w", "D": "d"}

def press_key(key_char):
    keyboard.press(key_char)
    keyboard.release(key_char)

sct = mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

def box_intersects(region, box):
    rx, ry, rw, rh = region
    x1, y1, x2, y2 = box
    return not (x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh)

last_key_pressed = None
last_key_time = 0


while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame, imgsz=IMGSZ, conf=CONFIDENCE_THRESHOLD, verbose=False)

    region_colliding = {k: False for k in REGIONS}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])

        if SHOW_WINDOW:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for direction, region in REGIONS.items():
            if not region_colliding[direction] and box_intersects(region, (x1, y1, x2, y2)):
                region_colliding[direction] = True

    key_pressed_this_frame = None
    current_time = time.time()
    for direction, collided in region_colliding.items():
        if collided:
            key = KEY_MAP[direction]
            if key != last_key_pressed or (current_time - last_key_time) >= RESET_DELAY:
                press_key(key)
                last_key_pressed = key
                last_key_time = current_time
                key_pressed_this_frame = key
            break 

    if key_pressed_this_frame and key_pressed_this_frame != last_key_pressed:
        last_key_pressed = key_pressed_this_frame
        last_key_time = current_time

    if SHOW_WINDOW:
        for direction, region in REGIONS.items():
            x, y, w, h = region
            color = (0, 0, 255) if region_colliding[direction] else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, direction, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        small_frame = cv2.resize(frame, (0, 0), fx=WINDOW_SCALE, fy=WINDOW_SCALE)
        cv2.imshow("YOLO Game Bot", small_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

if SHOW_WINDOW:
    cv2.destroyAllWindows()