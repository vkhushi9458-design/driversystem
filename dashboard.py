
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time
import math
import tkinter as tk

from pathlib import Path
from collections import deque
from datetime import datetime
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "driver_monitor.db"

EAR_CLOSE_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.24
EYES_CLOSED_SECONDS = 1.5

MOUTH_OPEN_THRESHOLD = 0.055
YAWN_SECONDS = 1.0

HEAD_DOWN_THRESHOLD = -145
HEAD_DOWN_SECONDS = 0.4

SMOOTHING_FRAMES = 3
CAMERA_INDEX = 0


# =========================================================
# DATABASE FUNCTIONS
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
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                """
                INSERT INTO events (event_type, event_time)
                VALUES (?, ?)
                """,
                (event_type, event_time)
            )

        print(f"[DATABASE] Saved {event_type} at {event_time}")

    except sqlite3.Error as error:
        print("Database save error:", error)


def get_event_counts():
    counts = {
        "EYES_CLOSED": 0,
        "YAWNING": 0,
        "HEAD_DOWN": 0
    }

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute("""
            SELECT event_type, COUNT(*)
            FROM events
            GROUP BY event_type
        """).fetchall()

    for event_type, count in rows:
        if event_type in counts:
            counts[event_type] = count

    return counts


def get_recent_events(limit=12):
    with sqlite3.connect(DATABASE_PATH) as connection:
        return connection.execute("""
            SELECT id, event_type, event_time
            FROM events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()


# =========================================================
# FACIAL MEASUREMENTS
# =========================================================

def distance(point1, point2):
    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1]
    )


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

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_mouth_ratio(landmarks, width, height):
    indices = [13, 14, 78, 308]
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

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
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    return float(angles[0])


# =========================================================
# DESKTOP DASHBOARD
# =========================================================

class DriverDashboard:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Driver Monitoring System")
        self.root.geometry("1250x820")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#edf1f5")

        create_database()

        # Detection state
        self.ear_history = deque(maxlen=SMOOTHING_FRAMES)
        self.eyes_closed_start = None
        self.yawn_start = None
        self.head_down_start = None

        # Prevent saving the same continuous warning repeatedly.
        self.active_events = set()

        self.last_ear = 0.0
        self.last_mouth_ratio = 0.0
        self.last_pitch = None

        self.running = True
        self.camera_error_shown = False
        self.photo = None

        self.left_eye_indices = [362, 385, 387, 263, 373, 380]
        self.right_eye_indices = [33, 160, 158, 133, 153, 144]

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        self.setup_styles()
        self.build_interface()
        self.refresh_history()

        if self.cap.isOpened():
            self.update_camera()
        else:
            self.camera_label.configure(
                text="Camera unavailable\nCheck your webcam connection.",
                image=""
            )
            self.status_label.configure(
                text="CAMERA UNAVAILABLE",
                foreground="#b42318"
            )

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # -----------------------------------------------------
    # STYLE
    # -----------------------------------------------------

    def setup_styles(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Treeview",
            font=("Segoe UI", 10),
            rowheight=27,
            background="white",
            fieldbackground="white"
        )

        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold")
        )

    # -----------------------------------------------------
    # BUILD THE INTERFACE
    # -----------------------------------------------------

    def build_interface(self):
        header = tk.Frame(self.root, bg="#172b4d", height=75)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="AI DRIVER MONITORING SYSTEM",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#172b4d"
        ).pack(side="left", padx=24, pady=18)

        tk.Label(
            header,
            text="LIVE MONITORING",
            font=("Segoe UI", 10, "bold"),
            fg="#b8e6d0",
            bg="#172b4d"
        ).pack(side="right", padx=24)

        main = tk.Frame(self.root, bg="#edf1f5")
        main.pack(fill="both", expand=True, padx=16, pady=14)

        # LEFT: camera and status
        left = tk.Frame(main, bg="#edf1f5")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        camera_card = tk.Frame(
            left, bg="#ffffff",
            highlightbackground="#d6dce5",
            highlightthickness=1
        )
        camera_card.pack(fill="both", expand=True)

        tk.Label(
            camera_card,
            text="LIVE CAMERA FEED",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#172b4d"
        ).pack(anchor="w", padx=14, pady=(12, 8))

        self.camera_label = tk.Label(
            camera_card,
            text="Starting camera...",
            bg="#111827",
            fg="white",
            font=("Segoe UI", 13),
            width=70,
            height=23
        )
        self.camera_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        status_card = tk.Frame(
            left, bg="white",
            highlightbackground="#d6dce5",
            highlightthickness=1
        )
        status_card.pack(fill="x", pady=(12, 0))

        tk.Label(
            status_card,
            text="DRIVER STATUS",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            fg="#667085"
        ).pack(anchor="w", padx=14, pady=(10, 3))

        self.status_label = tk.Label(
            status_card,
            text="INITIALIZING",
            font=("Segoe UI", 18, "bold"),
            bg="white",
            fg="#16803c"
        )
        self.status_label.pack(anchor="w", padx=14, pady=(0, 10))

        # RIGHT: measurements and history
        right = tk.Frame(main, bg="#edf1f5", width=390)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        metrics_card = tk.Frame(
            right, bg="white",
            highlightbackground="#d6dce5",
            highlightthickness=1
        )
        metrics_card.pack(fill="x")

        tk.Label(
            metrics_card,
            text="LIVE MEASUREMENTS",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#172b4d"
        ).pack(anchor="w", padx=14, pady=(12, 8))

        self.ear_label = self.add_metric(metrics_card, "Eye Aspect Ratio (EAR)")
        self.mouth_label = self.add_metric(metrics_card, "Mouth Ratio")
        self.pitch_label = self.add_metric(metrics_card, "Head Pitch")

        counts_card = tk.Frame(
            right, bg="white",
            highlightbackground="#d6dce5",
            highlightthickness=1
        )
        counts_card.pack(fill="x", pady=(12, 0))

        tk.Label(
            counts_card,
            text="SAVED WARNING COUNTS",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#172b4d"
        ).pack(anchor="w", padx=14, pady=(12, 8))

        self.eye_count = self.add_metric(counts_card, "Eyes Closed Events")
        self.yawn_count = self.add_metric(counts_card, "Yawning Events")
        self.head_count = self.add_metric(counts_card, "Head Down Events")

        history_card = tk.Frame(
            right, bg="white",
            highlightbackground="#d6dce5",
            highlightthickness=1
        )
        history_card.pack(fill="both", expand=True, pady=(12, 0))

        tk.Label(
            history_card,
            text="RECENT EVENT HISTORY",
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg="#172b4d"
        ).pack(anchor="w", padx=12, pady=(12, 8))

        columns = ("id", "event", "time")

        self.event_table = ttk.Treeview(
            history_card,
            columns=columns,
            show="headings",
            height=9
        )

        self.event_table.heading("id", text="ID")
        self.event_table.heading("event", text="Event")
        self.event_table.heading("time", text="Timestamp")

        self.event_table.column("id", width=42, anchor="center")
        self.event_table.column("event", width=125, anchor="w")
        self.event_table.column("time", width=170, anchor="w")

        scrollbar = ttk.Scrollbar(
            history_card,
            orient="vertical",
            command=self.event_table.yview
        )
        self.event_table.configure(yscrollcommand=scrollbar.set)

        self.event_table.pack(
            side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10)
        )
        scrollbar.pack(side="right", fill="y", pady=(0, 10), padx=(0, 8))

        footer = tk.Label(
            self.root,
            text=f"Database: {DATABASE_PATH}     |     Press Q or click Exit to close",
            font=("Segoe UI", 9),
            bg="#172b4d",
            fg="white",
            anchor="w",
            padx=14,
            pady=8
        )
        footer.pack(fill="x")

        exit_button = tk.Button(
            header,
            text="Exit",
            command=self.close_app,
            font=("Segoe UI", 10, "bold"),
            bg="#b42318",
            fg="white",
            activebackground="#912018",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=6
        )
        exit_button.pack(side="right", padx=(0, 16), pady=16)

    def add_metric(self, parent, title):
        row = tk.Frame(parent, bg="white")
        row.pack(fill="x", padx=14, pady=4)

        tk.Label(
            row,
            text=title,
            font=("Segoe UI", 10),
            bg="white",
            fg="#667085"
        ).pack(side="left")

        value_label = tk.Label(
            row,
            text="--",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#172b4d"
        )
        value_label.pack(side="right")

        return value_label

    # -----------------------------------------------------
    # CAMERA AND DETECTION
    # -----------------------------------------------------

    def update_camera(self):
        if not self.running:
            return

        success, frame = self.cap.read()

        if success:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            current_events = set()
            current_time = time.time()

            ear_value = 0.0
            mouth_ratio = 0.0
            head_pitch = None

            status_messages = []

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Eye closure
                left_ear = calculate_ear(
                    landmarks, self.left_eye_indices, width, height
                )
                right_ear = calculate_ear(
                    landmarks, self.right_eye_indices, width, height
                )

                raw_ear = (left_ear + right_ear) / 2.0
                self.ear_history.append(raw_ear)

                ear_value = sum(self.ear_history) / len(self.ear_history)

                if ear_value < EAR_CLOSE_THRESHOLD:
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = current_time

                    if current_time - self.eyes_closed_start >= EYES_CLOSED_SECONDS:
                        current_events.add("EYES_CLOSED")
                        status_messages.append("EYES CLOSED")

                elif ear_value > EAR_OPEN_THRESHOLD:
                    self.eyes_closed_start = None

                # Yawning
                mouth_ratio = calculate_mouth_ratio(
                    landmarks, width, height
                )

                if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                    if self.yawn_start is None:
                        self.yawn_start = current_time

                    if current_time - self.yawn_start >= YAWN_SECONDS:
                        current_events.add("YAWNING")
                        status_messages.append("YAWNING")

                else:
                    self.yawn_start = None

                # Head down
                head_pitch = calculate_head_pitch(
                    landmarks, width, height
                )

                if head_pitch is not None:
                    if head_pitch < HEAD_DOWN_THRESHOLD:
                        if self.head_down_start is None:
                            self.head_down_start = current_time

                        if current_time - self.head_down_start >= HEAD_DOWN_SECONDS:
                            current_events.add("HEAD_DOWN")
                            status_messages.append("HEAD DOWN")

                    else:
                        self.head_down_start = None

                # Draw useful facial landmarks
                for index in (
                    self.left_eye_indices
                    + self.right_eye_indices
                    + [13, 14, 78, 308]
                ):
                    point = (
                        int(landmarks[index].x * width),
                        int(landmarks[index].y * height)
                    )
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

            else:
                self.eyes_closed_start = None
                self.yawn_start = None
                self.head_down_start = None
                self.ear_history.clear()

            # Save each warning only when it becomes active.
            for event_type in current_events:
                if event_type not in self.active_events:
                    save_event(event_type)

            self.active_events = current_events.copy()

            # Measurements
            self.last_ear = ear_value
            self.last_mouth_ratio = mouth_ratio
            self.last_pitch = head_pitch

            self.ear_label.configure(text=f"{ear_value:.3f}")
            self.mouth_label.configure(text=f"{mouth_ratio:.3f}")
            self.pitch_label.configure(
                text=f"{head_pitch:.1f}°" if head_pitch is not None else "--"
            )

            # Status and on-camera warnings
            if status_messages:
                status_text = "WARNING: " + ", ".join(status_messages)
                status_color = "#b42318"
                overlay_color = (0, 0, 255)
            elif results.multi_face_landmarks:
                status_text = "NORMAL"
                status_color = "#16803c"
                overlay_color = (0, 200, 0)
            else:
                status_text = "NO FACE DETECTED"
                status_color = "#b54708"
                overlay_color = (0, 165, 255)

            self.status_label.configure(
                text=status_text,
                foreground=status_color
            )

            cv2.putText(
                frame,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                overlay_color,
                2
            )

            # Display the camera image inside Tkinter.
            display_width = max(self.camera_label.winfo_width(), 640)
            display_height = max(self.camera_label.winfo_height(), 360)

            scale = min(
                display_width / width,
                display_height / height
            )

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            frame = cv2.resize(frame, (new_width, new_height))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=image)

            self.camera_label.configure(image=self.photo, text="")

        else:
            self.camera_label.configure(
                text="Could not read from camera.",
                image=""
            )

        # Schedule the next frame without blocking the interface.
        self.root.after(10, self.update_camera)

    # -----------------------------------------------------
    # DATABASE DASHBOARD REFRESH
    # -----------------------------------------------------

    def refresh_history(self):
        if not self.running:
            return

        try:
            counts = get_event_counts()

            self.eye_count.configure(
                text=str(counts["EYES_CLOSED"])
            )
            self.yawn_count.configure(
                text=str(counts["YAWNING"])
            )
            self.head_count.configure(
                text=str(counts["HEAD_DOWN"])
            )

            recent_events = get_recent_events(12)

            # Preserve the current scroll position where possible.
            for item in self.event_table.get_children():
                self.event_table.delete(item)

            for event_id, event_type, event_time in recent_events:
                self.event_table.insert(
                    "",
                    "end",
                    values=(event_id, event_type, event_time)
                )

        except sqlite3.Error as error:
            print("Database refresh error:", error)

        self.root.after(2000, self.refresh_history)

    # -----------------------------------------------------
    # CLEAN EXIT
    # -----------------------------------------------------

    def close_app(self):
        if not self.running:
            return

        self.running = False

        try:
            self.cap.release()
        except Exception:
            pass

        try:
            self.face_mesh.close()
        except Exception:
            pass

        self.root.destroy()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = DriverDashboard(root)
    root.mainloop()
