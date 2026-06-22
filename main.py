import cv2
import numpy as np
import time
from ultralytics import YOLO

# CONFIG

MODEL_NAME = "yolov8n.pt"

CONFIDENCE = 0.35

HORIZONTAL_RATIO = 1.3
MOVEMENT_THRESHOLD = 40
SLEEP_TIME_THRESHOLD = 5

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# CAMERA SEARCH

def find_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        
        if cap.isOpened():
            ret, frame = cap.read()
            
            if ret and frame is not None:
                cap.release()
                return i
        cap.release()
    return None


camera_index = find_camera()

if camera_index is None:
    raise Exception("No camera found")


# CAMERA

cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

if not cap.isOpened():
    raise Exception(f"Failed to open camera {camera_index}")


# YOLO

model = YOLO(MODEL_NAME)

# TRACKING

people = {}
next_person_id = 1

# MAIN LOOP

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    current_time = time.time()

    results = model(
        frame,
        verbose=False
    )

    # DETECTION

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])

            if cls != 0:
                continue

            conf = float(box.conf[0])

            if conf < CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            ratio = width / height

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
 
            person_id = None

            # TRACK MATCH

            for pid, pdata in people.items():
                px, py = pdata["center"]
                dist = np.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                if dist < 100:
                    person_id = pid
                    break

            if person_id is None:
                person_id = next_person_id
                next_person_id += 1
                people[person_id] = {
                    "center": (cx, cy),
                    "horizontal_start": None,
                    "last_seen": current_time
                }

            pdata = people[person_id]
            old_x, old_y = pdata["center"]
            movement = np.sqrt(
                (cx - old_x) ** 2 +
                (cy - old_y) ** 2
            )

            pdata["center"] = (cx, cy)
            pdata["last_seen"] = current_time
            horizontal = ratio > HORIZONTAL_RATIO
            still = movement < MOVEMENT_THRESHOLD
            sleeping = False

            if horizontal and still:

                if pdata["horizontal_start"] is None:
                    pdata["horizontal_start"] = current_time

                elapsed = (
                    current_time -
                    pdata["horizontal_start"]
                )

                if elapsed >= SLEEP_TIME_THRESHOLD:
                    sleeping = True

            else:

                pdata["horizontal_start"] = None
                elapsed = 0

            # VISUALIZATION

            if sleeping:

                color = (0, 255, 0)
                status = "SLEEPING"

            elif horizontal:

                color = (0, 255, 255)
                status = "HORIZONTAL"

            else:

                color = (0, 165, 255)
                status = "PERSON"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text_y = y1 - 10

            cv2.putText(
                frame,
                f"ID:{person_id}",
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            cv2.putText(
                frame,
                status,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            cv2.putText(
                frame,
                f"C:{conf:.2f}",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"R:{ratio:.2f}",
                (x1, y2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"M:{movement:.1f}",
                (x1, y2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"T:{elapsed:.1f}",
                (x1, y2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    # CLEANUP

    remove_ids = []

    for pid, pdata in people.items():

        if current_time - pdata["last_seen"] > 3:
            remove_ids.append(pid)

    for pid in remove_ids:
        del people[pid]

    cv2.imshow(
        "NapSee Debug",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()