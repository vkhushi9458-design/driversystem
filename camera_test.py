
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam could not be opened.")
    raise SystemExit

print("Webcam started! Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read camera frame.")
        break

    cv2.imshow("Driver Monitoring System - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
