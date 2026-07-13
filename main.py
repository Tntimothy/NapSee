import ctypes
import ctypes.wintypes as wt

ctypes.windll.user32.SetProcessDPIAware()

# WIN32 BINDINGS

_u = ctypes.windll.user32
_g = ctypes.windll.gdi32

_u.IsWindowVisible.argtypes        = [wt.HWND]
_u.IsWindowVisible.restype         = wt.BOOL
_u.GetWindowTextLengthW.argtypes   = [wt.HWND]
_u.GetWindowTextLengthW.restype    = ctypes.c_int
_u.GetWindowTextW.argtypes         = [wt.HWND, wt.LPWSTR, ctypes.c_int]
_u.GetWindowTextW.restype          = ctypes.c_int
_u.GetWindowRect.argtypes          = [wt.HWND, ctypes.POINTER(wt.RECT)]
_u.GetWindowRect.restype           = wt.BOOL
_u.GetDC.argtypes                  = [wt.HWND]
_u.GetDC.restype                   = wt.HDC
_u.ReleaseDC.argtypes              = [wt.HWND, wt.HDC]
_u.ReleaseDC.restype               = ctypes.c_int
_u.PrintWindow.argtypes            = [wt.HWND, wt.HDC, ctypes.c_uint]
_u.PrintWindow.restype             = wt.BOOL
_u.FindWindowW.argtypes            = [wt.LPCWSTR, wt.LPCWSTR]
_u.FindWindowW.restype             = wt.HWND
_u.SetWindowPos.argtypes           = [wt.HWND, wt.HWND, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_uint]
_u.SetWindowPos.restype            = wt.BOOL

_g.CreateCompatibleDC.argtypes     = [wt.HDC]
_g.CreateCompatibleDC.restype      = wt.HDC
_g.CreateCompatibleBitmap.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
_g.CreateCompatibleBitmap.restype  = wt.HBITMAP
_g.SelectObject.argtypes           = [wt.HDC, wt.HGDIOBJ]
_g.SelectObject.restype            = wt.HGDIOBJ
_g.GetDIBits.argtypes              = [wt.HDC, wt.HBITMAP, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_uint]
_g.GetDIBits.restype               = ctypes.c_int
_g.DeleteObject.argtypes           = [wt.HGDIOBJ]
_g.DeleteObject.restype            = wt.BOOL
_g.DeleteDC.argtypes               = [wt.HDC]
_g.DeleteDC.restype                = wt.BOOL

import cv2
import numpy as np
import time
import threading

from ultralytics import YOLO
from pathlib import Path

# CONFIG

PROJECT_DIR = Path(__file__).resolve().parent
#DETECTOR_MODEL = PROJECT_DIR / "secondrun" / "visdrone_person" / "weights" / "best.pt"
DETECTOR_MODEL = PROJECT_DIR / "yolov8m.pt"
POSE_MODEL = PROJECT_DIR / "yolov8n-pose.pt"
CONFIDENCE = 0.12
BODY_ANGLE_THRESHOLD = 70
MOVEMENT_THRESHOLD = 80
POSSIBLE_SLEEP_TIME = 3
SLEEP_TIME = 6
MIN_AREA = 50
OCCUPANCY_THRESHOLD = 350
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
TILE_ROWS = 1
TILE_COLS = 1
WINDOW_NAME = "NapSee Window Capture"

# PERFORMANCE CONFIG

# I will try to edit this for actual cctvs
TILE_IMGSZ = 640
TARGET_FPS = 5 
MATCH_DIST = 100

# SKELETON

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# BITMAPINFOHEADER

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize",          wt.DWORD),
        ("biWidth",         wt.LONG),
        ("biHeight",        wt.LONG),
        ("biPlanes",        wt.WORD),
        ("biBitCount",      wt.WORD),
        ("biCompression",   wt.DWORD),
        ("biSizeImage",     wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed",       wt.DWORD),
        ("biClrImportant",  wt.DWORD),
    ]

# WINDOW ENUMERATION

WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

entries = []

def _enum_cb(hwnd, _):

    if not _u.IsWindowVisible(hwnd):
        return True

    n = _u.GetWindowTextLengthW(hwnd)

    if n == 0:
        return True

    buf = ctypes.create_unicode_buffer(n + 1)
    _u.GetWindowTextW(hwnd, buf, n + 1)
    title = buf.value.strip()

    if not title or WINDOW_NAME in title or "NapSee" in title:
        return True

    rect = wt.RECT()
    _u.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right  - rect.left
    h = rect.bottom - rect.top

    if w > 100 and h > 100:
        entries.append((title, w, h, hwnd))

    return True

print("\nOPEN WINDOWS:\n")

_cb = WNDENUMPROC(_enum_cb)
_u.EnumWindows(_cb, 0)

for i, (title, w, h, _) in enumerate(entries):
    print(f"[{i}]  {title}  ({w}x{h})")

choice = int(input("\nSelect window number: "))

target_title, _, _, target_hwnd = entries[choice]

print(f"\nCapturing: {target_title}\n")

# HELPERS

_mdc = None
_bmp = None
_cap_buf = None
_cap_w = 0
_cap_h = 0

_bih = BITMAPINFOHEADER()
_bih.biSize        = ctypes.sizeof(BITMAPINFOHEADER)
_bih.biPlanes      = 1
_bih.biBitCount    = 32
_bih.biCompression = 0


def get_dims(hwnd):

    rect = wt.RECT()

    if _u.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.right - rect.left, rect.bottom - rect.top

    return None


def _rebuild_capture_resources(hwnd, w, h):

    global _mdc, _bmp, _cap_buf, _cap_w, _cap_h

    if _mdc is not None:
        _g.DeleteObject(_bmp)
        _g.DeleteDC(_mdc)

    hdc = _u.GetDC(hwnd)
    _mdc = _g.CreateCompatibleDC(hdc)
    _bmp = _g.CreateCompatibleBitmap(hdc, w, h)
    _g.SelectObject(_mdc, _bmp)
    _u.ReleaseDC(hwnd, hdc)

    _cap_buf = (ctypes.c_uint8 * (w * h * 4))()
    _cap_w = w
    _cap_h = h
    _bih.biWidth  = w
    _bih.biHeight = h


def capture_window(hwnd, w, h):

    if w != _cap_w or h != _cap_h:
        _rebuild_capture_resources(hwnd, w, h)

    _u.PrintWindow(hwnd, _mdc, 2)
    _g.GetDIBits(_mdc, _bmp, 0, h, _cap_buf, ctypes.byref(_bih), 0)

    img = np.frombuffer(_cap_buf, dtype=np.uint8).reshape(h, w, 4)
    return img[::-1, :, :3].copy()


def set_topmost():

    hwnd = _u.FindWindowW(None, WINDOW_NAME)

    if hwnd:
        _u.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)


# THREADING

shared_frame = None
frame_lock   = threading.Lock()
stop_event   = threading.Event()

blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
cv2.putText(
    blank,
    "NapSee loading...",
    (DISPLAY_WIDTH // 2 - 160, DISPLAY_HEIGHT // 2),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
)
shared_frame = blank


def detection_loop():

    global shared_frame

    detector = YOLO(str(DETECTOR_MODEL))
    pose_model = YOLO(str(POSE_MODEL))

    detector.to("cpu")
    pose_model.to("cpu")

    people = {}
    next_id = 1

    while not stop_event.is_set():

        frame_start = time.time()

        dims = get_dims(target_hwnd)

        if dims is None:
            time.sleep(0.05)
            continue

        cap_w, cap_h = dims

        if cap_w < 100:
            time.sleep(0.05)
            continue

        try:
            frame_raw = capture_window(target_hwnd, cap_w, cap_h)
        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame_raw, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        current_time = time.time()

        h, w = frame.shape[:2]

        tile_w = w // TILE_COLS
        tile_h = h // TILE_ROWS

        detections = []

        try:

            for row in range(TILE_ROWS):

                for col in range(TILE_COLS):

                    x_offset = col * tile_w
                    y_offset = row * tile_h

                    tile = frame[
                        y_offset:y_offset + tile_h,
                        x_offset:x_offset + tile_w
                    ]

                    results = detector.predict(
                        tile,
                        imgsz=TILE_IMGSZ,
                        conf=CONFIDENCE,
                        device="cpu",
                        verbose=False
                    )

                    for result in results:

                        if result.boxes is None:
                            continue

                        for box in result.boxes:

                            cls = int(box.cls[0])

                            if cls not in (0, 1):
                                continue

                            if float(box.conf[0]) < CONFIDENCE:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            x1 += x_offset
                            x2 += x_offset
                            y1 += y_offset
                            y2 += y_offset

                            pw = x2 - x1
                            ph = y2 - y1

                            if pw * ph < MIN_AREA:
                                continue

                            detections.append({
                                "box": (x1, y1, x2, y2),
                                "width": pw,
                                "height": ph
                            })

        except Exception as e:
            print(f"Inference error: {e}")
            time.sleep(0.1)
            continue

        n_people = 0
        n_suspects = 0

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            pw = detection["width"]
            ph = detection["height"]

            n_people += 1

            pad = 30

            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(w, x2 + pad)
            crop_y2 = min(h, y2 + pad)

            crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            kp = None

            try:

                pose_results = pose_model.predict(
                    crop,
                    imgsz=512,
                    conf=0.25,
                    device="cpu",
                    verbose=False
                )

                if (
                    len(pose_results)
                    and pose_results[0].keypoints is not None
                    and len(pose_results[0].keypoints.xy)
                ):

                    kp = pose_results[0].keypoints.xy.cpu().numpy()[0]

                    kp[:, 0] += crop_x1
                    kp[:, 1] += crop_y1

            except Exception:
                kp = None

            ratio = pw / max(ph, 1)
            occupancy_score = ratio * pw

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            person_id = None
            best_dist = MATCH_DIST

            for pid, pdata in people.items():

                px, py = pdata["center"]

                dist = np.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                if dist < best_dist:
                    best_dist = dist
                    person_id = pid

            if person_id is None:

                person_id = next_id
                next_id += 1

                people[person_id] = {
                    "center": (cx, cy),
                    "start_time": None,
                    "last_seen": current_time,
                }

            pdata = people[person_id]

            old_x, old_y = pdata["center"]

            movement = np.sqrt(
                (cx - old_x) ** 2 +
                (cy - old_y) ** 2
            )

            pdata["center"] = (cx, cy)
            pdata["last_seen"] = current_time

            angle = 90
            horizontal_score = 0

            if kp is not None:

                try:

                    sx = (kp[5][0] + kp[6][0]) / 2
                    sy = (kp[5][1] + kp[6][1]) / 2

                    hx = (kp[11][0] + kp[12][0]) / 2
                    hy = (kp[11][1] + kp[12][1]) / 2

                    angle = abs(
                        np.degrees(
                            np.arctan2(
                                hy - sy,
                                hx - sx
                            )
                        )
                    )

                    if angle < BODY_ANGLE_THRESHOLD:
                        horizontal_score += 3

                except Exception:
                    pass

            if ratio > 1.2:
                horizontal_score += 2

            if movement < MOVEMENT_THRESHOLD:
                horizontal_score += 1

            if occupancy_score > OCCUPANCY_THRESHOLD:
                horizontal_score += 2

            lying = horizontal_score >= 4

            if lying:

                if pdata["start_time"] is None:
                    pdata["start_time"] = current_time

                elapsed = current_time - pdata["start_time"]

                if elapsed >= SLEEP_TIME:
                    status = "BENCH MISUSE"
                    color = (0, 255, 0)

                elif elapsed >= POSSIBLE_SLEEP_TIME:
                    status = "POSSIBLE"
                    color = (0, 255, 255)

                else:
                    status = "LYING"
                    color = (255, 255, 0)

            else:

                pdata["start_time"] = None
                elapsed = 0
                status = "PERSON"
                color = (100, 100, 100)

            thickness = 1 if status == "PERSON" else 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            if status != "PERSON":
                n_suspects += 1

            if kp is not None:

                for x, y in kp:

                    if x > 0 and y > 0:
                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            3,
                            (0, 255, 255),
                            -1
                        )

                for p1, p2 in SKELETON:

                    x1s, y1s = kp[p1]
                    x2s, y2s = kp[p2]

                    if (
                        x1s > 0 and
                        y1s > 0 and
                        x2s > 0 and
                        y2s > 0
                    ):
                        cv2.line(
                            frame,
                            (int(x1s), int(y1s)),
                            (int(x2s), int(y2s)),
                            (255, 255, 0),
                            2
                        )

            cv2.putText(
                frame,
                f"ID:{person_id} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            cv2.putText(
                frame,
                f"A:{angle:.0f}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"S:{horizontal_score}",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"O:{occupancy_score:.0f}",
                (x1, y2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

            cv2.putText(
                frame,
                f"T:{elapsed:.1f}s",
                (x1, y2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        cv2.putText(
            frame,
            f"People: {n_people}   Suspects: {n_suspects}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        remove_ids = []

        for pid, pdata in people.items():

            if current_time - pdata["last_seen"] > 3:
                remove_ids.append(pid)

        for pid in remove_ids:
            del people[pid]

        with frame_lock:
            shared_frame = frame.copy()

        spent = time.time() - frame_start

        remaining = (1.0 / TARGET_FPS) - spent

        if remaining > 0:
            time.sleep(remaining)

# MAIN LOOP

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

worker = threading.Thread(target=detection_loop, daemon=True)
worker.start()

topmost_set = False

while not stop_event.is_set():

    with frame_lock:
        frame_to_show = shared_frame

    cv2.imshow(WINDOW_NAME, frame_to_show)

    if not topmost_set:
        set_topmost()
        topmost_set = True

    if cv2.waitKey(15) == 27:
        stop_event.set()

worker.join(timeout=2)
cv2.destroyAllWindows()