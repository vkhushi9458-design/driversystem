
import cv2
import mediapipe as mp
import math
import time
from collections import deque
from statistics import median

# ---------------- SETTINGS ----------------
YAWN_THRESHOLD = 0.055
YAWN_SECONDS = 1.0
SMOOTHING_FRAMES = 5

# MediaPipe mouth landmark indices
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
UPPER_LIP = 13
LOWER_LIP = 14

mp_face_mesh = mp.solutions.face_mesh


def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def mouth_ratio(landmarks, width, height):
    # Get mouth landmark coordinates
    indices = [
        MOUTH_LEFT,
        MOUTH_RIGHT,
        UPPER_LIP,
        LOWER_LIP
    ]

    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    left = points[0]
    right = points[1]
    upper = points[2]
    lower = points[3]

    vertical_opening = distance(upper, lower)
    horizontal_width = distance(left, right)

    if horizontal_width == 0:
        return 0.0

    return vertical_opening / horizontal_width


# ---------------- INITIALIZE ----------------
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    face_mesh.close()
    raise RuntimeError("Could not open webcam.")

ratio_history = deque(maxlen=SMOOTHING_FRAMES)
mouth_open_start = None

print("Yawning detection started.")
print("Press Q to quit.")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        status = "No face detected"
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            raw_ratio = mouth_ratio(
                landmarks,
                width,
                height
            )

            ratio_history.append(raw_ratio)
            smooth_ratio = median(ratio_history)

            # Detect sustained mouth opening
            if smooth_ratio > YAWN_THRESHOLD:
                if mouth_open_start is None:
                    mouth_open_start = time.monotonic()

                duration = time.monotonic() - mouth_open_start

                if duration >= YAWN_SECONDS:
                    status = "POSSIBLE YAWN DETECTED"
                    color = (0, 0, 255)
                else:
                    status = "Mouth open..."
                    color = (0, 165, 255)

            else:
                mouth_open_start = None
                status = "Mouth normal"
                color = (0, 200, 0)

            # Display raw ratio
            cv2.putText(
                frame,
                f"Raw mouth ratio: {raw_ratio:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # Display smoothed ratio
            cv2.putText(
                frame,
                f"Smoothed ratio: {smooth_ratio:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # Display threshold
            cv2.putText(
                frame,
                f"Threshold: {YAWN_THRESHOLD:.3f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

        else:
            ratio_history.clear()
            mouth_open_start = None

        # Display detection status
        cv2.putText(
            frame,
            status,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow(
            "Driver Monitoring - Yawn Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()
