
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import math
import threading
import os

from pathlib import Path
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, render_template

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", BASE_DIR / "driver_monitor.db"))

EAR_CLOSE_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.24
EYES_CLOSED_SECONDS = 1.5

MOUTH_OPEN_THRESHOLD = 0.055
YAWN_SECONDS = 1.0

HEAD_DOWN_THRESHOLD = -145
HEAD_DOWN_SECONDS = 0.4

SMOOTHING_FRAMES = 3

CAMERA_URL = "http://10.17.5.51:8080/video"

app = Flask(__name__)

# =========================================================
# SHARED DETECTION STATE
# =========================================================

camera = None
face_mesh = None
camera_lock = threading.Lock()

ear_history = deque(maxlen=SMOOTHING_FRAMES)
eyes_closed_start = None
yawn_start = None
head_down_start = None
active_events = set()

left_eye_indices = [362, 385, 387, 263, 373, 380]
right_eye_indices = [33, 160, 158, 133, 153, 144]

state = {
    "camera": "Waiting for browser",
    "face_detected": False,
    "status": "Waiting for camera",
    "ear": 0.0,
    "mouth_ratio": 0.0,
    "pitch": None,
    "warnings": [],
    "updated_at": None,
}

# =========================================================
# DATABASE
# =========================================================

def create_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL
            )
        """)


def save_event(event_type):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(DATABASE_PATH, timeout=5) as connection:
            connection.execute(
                """
                INSERT INTO events (event_type, event_time)
                VALUES (?, ?)
                """,
                (event_type, event_time),
            )
    except sqlite3.Error as error:
        print("[DATABASE ERROR]", error)


def get_history():
    counts = {
        "EYES_CLOSED": 0,
        "YAWNING": 0,
        "HEAD_DOWN": 0,
    }

    with sqlite3.connect(DATABASE_PATH, timeout=5) as connection:
        rows = connection.execute("""
            SELECT event_type, COUNT(*)
            FROM events
            GROUP BY event_type
        """).fetchall()

        for event_type, count in rows:
            if event_type in counts:
                counts[event_type] = count

        recent = connection.execute("""
            SELECT id, event_type, event_time
            FROM events
            ORDER BY id DESC
            LIMIT 12
        """).fetchall()

    return {
        "counts": counts,
        "events": [
            {"id": row[0], "type": row[1], "time": row[2]}
            for row in recent
        ],
    }


# =========================================================
# FACIAL MEASUREMENTS
# =========================================================

def distance(point1, point2):
    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1],
    )


def calculate_ear(landmarks, indices, width, height):
    points = [
        (
            int(landmarks[i].x * width),
            int(landmarks[i].y * height),
        )
        for i in indices
    ]

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_mouth_ratio(landmarks, width, height):
    indices = [13, 14, 78, 308]

    points = [
        (
            int(landmarks[i].x * width),
            int(landmarks[i].y * height),
        )
        for i in indices
    ]

    mouth_height = distance(points[0], points[1])
    mouth_width = distance(points[2], points[3])

    if mouth_width == 0:
        return 0.0

    return mouth_height / mouth_width


def calculate_head_pitch(landmarks, width, height):
    indices = [1, 152, 33, 263, 61, 291]

    image_points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in indices
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    return float(angles[0])


# =========================================================
# CAMERA AND AI PROCESSING
# =========================================================

def generate_frames():
    global camera, face_mesh
    global eyes_closed_start, yawn_start, head_down_start
    global active_events

    try:
        # Open the webcam only when a browser requests the stream.
        with camera_lock:
            if camera is None or not camera.isOpened():
                if camera is not None:
                    camera.release()

                camera = cv2.VideoCapture(CAMERA_URL)

                if not camera.isOpened():
                    state["camera"] = "Unavailable"
                    state["status"] = "Phone IP webcam unavailable"
                    camera.release()
                    camera = None
                    return

            if face_mesh is None:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

        while True:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break

                success, frame = camera.read()

                if not success:
                    state["camera"] = "Frame read failed"
                    continue

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

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # Eye closure detection
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

                        if current_time - eyes_closed_start >= EYES_CLOSED_SECONDS:
                            current_events.add("EYES_CLOSED")
                            status_messages.append("EYES CLOSED")
                    elif ear_value > EAR_OPEN_THRESHOLD:
                        eyes_closed_start = None

                    # Yawning detection
                    mouth_ratio = calculate_mouth_ratio(
                        landmarks, width, height
                    )

                    if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                        if yawn_start is None:
                            yawn_start = current_time

                        if current_time - yawn_start >= YAWN_SECONDS:
                            current_events.add("YAWNING")
                            status_messages.append("YAWNING")
                    else:
                        yawn_start = None

                    # Head-down detection
                    head_pitch = calculate_head_pitch(
                        landmarks, width, height
                    )

                    if head_pitch is not None:
                        if head_pitch < HEAD_DOWN_THRESHOLD:
                            if head_down_start is None:
                                head_down_start = current_time

                            if current_time - head_down_start >= HEAD_DOWN_SECONDS:
                                current_events.add("HEAD_DOWN")
                                status_messages.append("HEAD DOWN")
                        else:
                            head_down_start = None

                    # Draw facial landmarks
                    for index in (
                        left_eye_indices
                        + right_eye_indices
                        + [13, 14, 78, 308]
                    ):
                        point = (
                            int(landmarks[index].x * width),
                            int(landmarks[index].y * height),
                        )
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)

                else:
                    eyes_closed_start = None
                    yawn_start = None
                    head_down_start = None
                    ear_history.clear()

                # Log each warning once when it first becomes active.
                for event_type in current_events:
                    if event_type not in active_events:
                        save_event(event_type)

                active_events = current_events.copy()

                if status_messages:
                    status_text = "WARNING: " + ", ".join(status_messages)
                    overlay_color = (0, 0, 255)
                elif results.multi_face_landmarks:
                    status_text = "NORMAL"
                    overlay_color = (0, 200, 0)
                else:
                    status_text = "NO FACE DETECTED"
                    overlay_color = (0, 165, 255)

                cv2.putText(
                    frame,
                    f"EAR: {ear_value:.3f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Mouth ratio: {mouth_ratio:.3f}",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                pitch_text = (
                    f"Head pitch: {head_pitch:.1f}"
                    if head_pitch is not None
                    else "Head pitch: unavailable"
                )

                cv2.putText(
                    frame,
                    pitch_text,
                    (15, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    status_text,
                    (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    overlay_color,
                    2,
                )

                state.update({
                    "camera": "Connected",
                    "face_detected": bool(results.multi_face_landmarks),
                    "status": status_text,
                    "ear": round(ear_value, 3),
                    "mouth_ratio": round(mouth_ratio, 3),
                    "pitch": round(head_pitch, 1) if head_pitch is not None else None,
                    "warnings": sorted(current_events),
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                })

                encoded, buffer = cv2.imencode(".jpg", frame)

            if encoded:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

    except GeneratorExit:
        # The browser closed its camera-stream connection.
        pass

    except Exception as error:
        print("[CAMERA ERROR]", error)
        state["camera"] = "Error"
        state["status"] = "Camera processing error"

# =========================================================
# WEBSITE
# =========================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Driver Monitoring</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef2f7;
    color: #172033;
}
header {
    background: #172b4d;
    color: white;
    padding: 22px;
}
header h1 { margin: 0 0 7px; font-size: 25px; }
header p { margin: 0; color: #d4deed; }
main { max-width: 1200px; width: 94%; margin: 22px auto; }
.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 9px #14213d10;
    min-width: 0;
}
.wide { grid-column: 1 / -1; }
h2 { font-size: 18px; margin-top: 0; }
.camera {
    display: block;
    width: 100%;
    height: auto;
    min-height: 160px;
    background: #111827;
    border-radius: 8px;
}
.metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}
.metric {
    padding: 12px;
    background: #f1f5f9;
    border-radius: 8px;
    overflow-wrap: anywhere;
}
.metric span { display: block; font-size: 12px; color: #64748b; }
.metric strong { display: block; font-size: 20px; margin-top: 8px; }
.status {
    font-weight: bold;
    padding: 9px 12px;
    border-radius: 7px;
    background: #dcfce7;
    color: #166534;
    display: inline-block;
}
.status.warning { background: #fee2e2; color: #991b1b; }
.status.waiting { background: #fef3c7; color: #92400e; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 8px; text-align: left; border-bottom: 1px solid #e2e8f0; }
th { background: #f8fafc; }
footer { text-align: center; color: #64748b; padding: 18px; font-size: 13px; }
@media(max-width: 650px) {
    header { padding: 18px 14px; }
    header h1 { font-size: 21px; }
    main { width: 94%; margin: 14px auto; }
    .grid { grid-template-columns: 1fr; }
    .wide { grid-column: auto; }
    .card { padding: 14px; }
    .metrics { grid-template-columns: 1fr; }
    .metric { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .metric strong { margin-top: 0; }
}
</style>
</head>
<body>
<header>
    <h1>AI Driver Monitoring System</h1>
    <p>Drowsiness detection and driver safety dashboard</p>
</header>

<main>
<div class="grid">
    <section class="card">
        <h2>Driver Status</h2>
        <p><span id="status" class="status waiting">Waiting for camera</span></p>
        <p>Camera: <strong id="camera">Waiting for connection</strong></p>
        <p>Face detected: <strong id="face">--</strong></p>
        <p>Last update: <strong id="updated">--</strong></p>
    </section>

    <section class="card">
        <h2>Live Measurements</h2>
        <div class="metrics">
            <div class="metric"><span>Eye ratio</span><strong id="ear">--</strong></div>
            <div class="metric"><span>Mouth ratio</span><strong id="mouth">--</strong></div>
            <div class="metric"><span>Head pitch</span><strong id="pitch">--</strong></div>
        </div>
    </section>

    <section class="card wide">
        <h2>Live Camera and AI Detection</h2>
        <img class="camera" src="/video_feed" alt="Live camera stream">
        <p>The camera and AI processing run on the computer hosting this website.</p>
    </section>

    <section class="card wide">
        <h2>Saved Warning Counts</h2>
        <div class="metrics">
            <div class="metric"><span>Eyes closed events</span><strong id="eyes_count">0</strong></div>
            <div class="metric"><span>Yawning events</span><strong id="yawn_count">0</strong></div>
            <div class="metric"><span>Head down events</span><strong id="head_count">0</strong></div>
        </div>
    </section>

    <section class="card wide">
        <h2>Recent Event History</h2>
        <div class="table-wrap">
        <table>
            <thead><tr><th>ID</th><th>Event</th><th>Date and time</th></tr></thead>
            <tbody id="events">
                <tr><td colspan="3">Loading events...</td></tr>
            </tbody>
        </table>
        </div>
    </section>
</div>
</main>
<footer>AI-Powered Driver Monitoring and Drowsiness Detection System</footer>

<script>
async function refreshDashboard() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const s = document.getElementById('status');
        s.textContent = data.status;
        s.className = 'status';

        if (data.warnings.length) {
            s.classList.add('warning');
        } else if (data.camera !== 'Connected') {
            s.classList.add('waiting');
        }

        document.getElementById('camera').textContent = data.camera;
        document.getElementById('face').textContent =
            data.face_detected ? 'Yes' : 'No';
        document.getElementById('ear').textContent = data.ear;
        document.getElementById('mouth').textContent = data.mouth_ratio;
        document.getElementById('pitch').textContent =
            data.pitch === null ? '--' : data.pitch + '°';
        document.getElementById('updated').textContent = data.updated_at || '--';

        const historyResponse = await fetch('/api/history');
        const history = await historyResponse.json();

        document.getElementById('eyes_count').textContent =
            history.counts.EYES_CLOSED;
        document.getElementById('yawn_count').textContent =
            history.counts.YAWNING;
        document.getElementById('head_count').textContent =
            history.counts.HEAD_DOWN;

        const tbody = document.getElementById('events');
        tbody.replaceChildren();

        if (!history.events.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 3;
            td.textContent = 'No warning events saved yet.';
            tr.appendChild(td);
            tbody.appendChild(tr);
        } else {
            for (const event of history.events) {
                const tr = document.createElement('tr');
                for (const value of [event.id, event.type, event.time]) {
                    const td = document.createElement('td');
                    td.textContent = value;
                    tr.appendChild(td);
                }
                tbody.appendChild(tr);
            }
        }
    } catch (error) {
        console.error('Dashboard refresh failed:', error);
    }
}

refreshDashboard();
setInterval(refreshDashboard, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    return jsonify(dict(state))


@app.route("/api/history")
def api_history():
    try:
        return jsonify(get_history())
    except sqlite3.Error as error:
        return jsonify({"error": str(error)}), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    create_database()
    print("Starting AI Driver Monitoring website...")
    print("Open http://127.0.0.1-:5000 in your browser.")

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        if camera is not None:
            camera.release()

        if face_mesh is not None:
            face_mesh.close()

        cv2.destroyAllWindows()
        print("Camera and AI system stopped.")
