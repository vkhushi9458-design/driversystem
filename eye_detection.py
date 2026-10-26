
import cv2
import mediapipe as mp
import math
import time
from collections import deque
from statistics import median

# ---------------- SETTINGS ----------------
CLOSE_THRESHOLD = 0.20
OPEN_THRESHOLD = 0.24
CLOSED_SECONDS = 1.5
SMOOTHING_FRAMES = 5

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal < 1:
        return None

    return (vertical1 + vertical2) / (2.0 * horizontal)


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

ear_history = deque(maxlen=SMOOTHING_FRAMES)
eyes_closed = False
closed_start = None

print("Improved eye detection started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "No face detected"
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            if left_ear is not None and right_ear is not None:
                # Average both eyes, then smooth across recent frames
                raw_ear = (left_ear + right_ear) / 2
                ear_history.append(raw_ear)
                ear = median(ear_history)

                # Hysteresis: use separate thresholds for closing/opening
                if not eyes_closed and ear < CLOSE_THRESHOLD:
                    eyes_closed = True
                    closed_start = time.monotonic()

                elif eyes_closed and ear > OPEN_THRESHOLD:
                    eyes_closed = False
                    closed_start = None

                if eyes_closed and closed_start is not None:
                    duration = time.monotonic() - closed_start

                    if duration >= CLOSED_SECONDS:
                        status = "DROWSINESS WARNING!"
                        color = (0, 0, 255)
                    else:
                        status = "Eyes possibly closed..."
                        color = (0, 165, 255)
                else:
                    status = "Eyes open"
                    color = (0, 200, 0)

                cv2.putText(
                    frame,
                    f"Left EAR: {left_ear:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Right EAR: {right_ear:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Smoothed EAR: {ear:.3f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

            else:
                ear_history.clear()
                eyes_closed = False
                closed_start = None

        else:
            # Reset if the face disappears
            ear_history.clear()
            eyes_closed = False
            closed_start = None

        cv2.putText(
            frame,
            status,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, 2
        )

        cv2.imshow("Improved Driver Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()
