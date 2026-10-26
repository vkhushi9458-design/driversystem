
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import math
from pathlib import Path
from collections import deque
from datetime import datetime


# =========================================================
# CONFIGURATION
# =========================================================

EAR_CLOSE_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.24
EYES_CLOSED_SECONDS = 1.5

MOUTH_OPEN_THRESHOLD = 0.055
YAWN_SECONDS = 1.0

HEAD_DOWN_THRESHOLD = -145
HEAD_DOWN_SECONDS = 0.4

SMOOTHING_FRAMES = 3

DATABASE_PATH = Path(__file__).resolve().parent / "driver_monitor.db"


# =========================================================
# DATABASE
# =========================================================

def create_database():
    """Create the database and events table if they do not exist."""

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_event(event_type):
    """Save one warning event to SQLite."""

    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        connection = sqlite3.connect(DATABASE_PATH)

        connection.execute(
            """
            INSERT INTO events (event_type, event_time)
            VALUES (?, ?)
            """,
            (event_type, event_time)
        )

        connection.commit()
        connection.close()

        print(f"[DATABASE] Saved event: {event_type} at {event_time}")

    except sqlite3.Error as error:
        print(f"[DATABASE ERROR] {error}")


# =========================================================
# FACIAL MEASUREMENT FUNCTIONS
# =========================================================

def calculate_distance(point1, point2):
    """Calculate the distance between two 2D points."""

    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1]
    )


def calculate_ear(landmarks, eye_indices, width, height):
    """Calculate the Eye Aspect Ratio (EAR)."""

    points = []

    for index in eye_indices:
        landmark = landmarks[index]

        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical_1 = calculate_distance(points[1], points[5])
    vertical_2 = calculate_distance(points[2], points[4])
    horizontal = calculate_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mouth_ratio(landmarks, width, height):
    """Measure mouth opening relative to mouth width."""

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_corner = landmarks[78]
    right_corner = landmarks[308]

    upper_point = (
        int(upper_lip.x * width),
        int(upper_lip.y * height)
    )

    lower_point = (
        int(lower_lip.x * width),
        int(lower_lip.y * height)
    )

    left_point = (
        int(left_corner.x * width),
        int(left_corner.y * height)
    )

    right_point = (
        int(right_corner.x * width),
        int(right_corner.y * height)
    )

    mouth_height = calculate_distance(upper_point, lower_point)
    mouth_width = calculate_distance(left_point, right_point)

    if mouth_width == 0:
        return 0.0

    return mouth_height / mouth_width


# =========================================================
# HEAD POSE ESTIMATION
# =========================================================

def calculate_head_pitch(landmarks, width, height):
    """
    Estimate head pitch using facial landmarks and solvePnP.

    Returns the pitch angle in degrees, or None if pose
    estimation fails.
    """

    landmark_indices = [1, 152, 33, 263, 61, 291]

    image_points = []

    for index in landmark_indices:
        landmark = landmarks[index]

        image_points.append([
            landmark.x * width,
            landmark.y * height
        ])

    image_points = np.array(image_points, dtype=np.float64)

    # Approximate 3D facial reference points.
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    # X-axis rotation is used as the pitch estimate.
    pitch = float(angles[0])

    return pitch


# =========================================================
# INITIALIZATION
# =========================================================

create_database()

mp_face_mesh = mp.solutions.face_mesh

left_eye_indices = [362, 385, 387, 263, 373, 380]
right_eye_indices = [33, 160, 158, 133, 153, 144]

ear_history = deque(maxlen=SMOOTHING_FRAMES)

eyes_closed_start = None
yawn_start = None
head_down_start = None

# Event types that were active in the previous frame.
active_events = set()

phone="http://10.17.5.51:8080/video"
camera = cv2.VideoCapture(phone)

if not camera.isOpened():
    print("ERROR: Could not open the camera.")
    print("Check that your webcam is connected and not being used elsewhere.")
    raise SystemExit(1)


# =========================================================
# MAIN MONITORING LOOP
# =========================================================

try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            current_time = time.time()
            current_events = set()

            ear_value = 0.0
            mouth_ratio = 0.0
            head_pitch = None

            status_messages = []

            # -------------------------------------------------
            # FACE DETECTION
            # -------------------------------------------------

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # ---------------------------------------------
                # EYE CLOSURE DETECTION
                # ---------------------------------------------

                left_ear = calculate_ear(
                    landmarks, left_eye_indices, width, height
                )

                right_ear = calculate_ear(
                    landmarks, right_eye_indices, width, height
                )

                raw_ear = (left_ear + right_ear) / 2.0

                ear_history.append(raw_ear)

                ear_value = sum(ear_history) / len(ear_history)

                if ear_value < EAR_CLOSE_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time

                    closed_duration = current_time - eyes_closed_start

                    if closed_duration >= EYES_CLOSED_SECONDS:
                        current_events.add("EYES_CLOSED")
                        status_messages.append("WARNING: EYES CLOSED")

                elif ear_value > EAR_OPEN_THRESHOLD:
                    eyes_closed_start = None

                # ---------------------------------------------
                # YAWN DETECTION
                # ---------------------------------------------

                mouth_ratio = calculate_mouth_ratio(
                    landmarks, width, height
                )

                if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = current_time

                    yawn_duration = current_time - yawn_start

                    if yawn_duration >= YAWN_SECONDS:
                        current_events.add("YAWNING")
                        status_messages.append("WARNING: YAWNING")

                else:
                    yawn_start = None

                # ---------------------------------------------
                # HEAD-DOWN DETECTION
                # ---------------------------------------------

                head_pitch = calculate_head_pitch(
                    landmarks, width, height
                )

                if head_pitch is not None:
                    if head_pitch < HEAD_DOWN_THRESHOLD:
                        if head_down_start is None:
                            head_down_start = current_time

                        head_down_duration = (
                            current_time - head_down_start
                        )

                        if head_down_duration >= HEAD_DOWN_SECONDS:
                            current_events.add("HEAD_DOWN")
                            status_messages.append("WARNING: HEAD DOWN")

                    else:
                        head_down_start = None

                # ---------------------------------------------
                # DRAW FACE LANDMARKS
                # ---------------------------------------------

                for index in (
                    left_eye_indices + right_eye_indices
                    + [13, 14, 78, 308]
                ):
                    landmark = landmarks[index]

                    point = (
                        int(landmark.x * width),
                        int(landmark.y * height)
                    )

                    cv2.circle(
                        frame,
                        point,
                        2,
                        (0, 255, 0),
                        -1
                    )

            else:
                # Reset timers when the face cannot be detected.
                eyes_closed_start = None
                yawn_start = None
                head_down_start = None
                ear_history.clear()

            # =================================================
            # DATABASE EVENT LOGGING — IMPORTANT FIX
            # =================================================
            #
            # Save an event only if it is active now but was
            # not active in the previous frame.
            #
            # Do not move this block inside a detection section.

            for event_type in current_events:
                if event_type not in active_events:
                    save_event(event_type)

            # Remember the current frame's active warnings.
            # This allows a later, separate warning to be logged.
            active_events = current_events.copy()

            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"EAR: {ear_value:.3f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Mouth ratio: {mouth_ratio:.3f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            pitch_text = (
                f"Head pitch: {head_pitch:.1f}"
                if head_pitch is not None
                else "Head pitch: unavailable"
            )

            cv2.putText(
                frame,
                pitch_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if status_messages:
                for index, message in enumerate(status_messages):
                    cv2.putText(
                        frame,
                        message,
                        (20, 135 + index * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
            else:
                cv2.putText(
                    frame,
                    "Status: NORMAL",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("AI Driver Monitoring System", frame)

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Driver monitoring system stopped.")
