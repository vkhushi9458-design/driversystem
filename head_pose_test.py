
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# 2D facial landmarks and corresponding approximate 3D face points
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

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

print("Head pose test started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "No face detected"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            image_points = np.array([
                (
                    landmarks[i].x * width,
                    landmarks[i].y * height
                )
                for i in LANDMARK_IDS
            ], dtype=np.float64)

            focal_length = width
            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            distortion = np.zeros((4, 1))

            success, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    MODEL_POINTS,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            )

            if success:
                rotation_matrix, _ = cv2.Rodrigues(
                    rotation_vector
                )

                angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                    rotation_matrix
                )

                pitch, yaw, roll = angles

                status = (
                    f"Pitch: {pitch:.1f}  "
                    f"Yaw: {yaw:.1f}  "
                    f"Roll: {roll:.1f}"
                )

                # Draw selected facial points
                for point in image_points:
                    cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        4,
                        (0, 255, 0),
                        -1
                    )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("Driver Monitoring - Head Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()
