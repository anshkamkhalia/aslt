import cv2 as cv
import os
import time

GESTURE = "I_AGREE"
# change to "NO", "HELLO", etc.
SAVE_DIR = f"data/{GESTURE}"
FPS = 30
VIDEO_LEN = 2  # seconds per clip
CAM_INDEX = 0  # your webcam

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv.VideoCapture(CAM_INDEX)
cap.set(cv.CAP_PROP_FPS, FPS)

count = len(os.listdir(SAVE_DIR))
print(f"Recording gesture: {GESTURE}")
print("Press 'r' to record, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    cv.putText(frame, f"Gesture: {GESTURE}", (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv.putText(frame, f"Clips saved: {count}", (10, 80),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(frame, "Press 'r' to record", (10, 120),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv.imshow("Gesture Recorder", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        filename = os.path.join(SAVE_DIR, f"{GESTURE.lower()}_{count}.mp4")
        print(f"Recording {filename}")

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(filename, fourcc, FPS,
                             (frame.shape[1], frame.shape[0]))

        start_time = time.time()
        while time.time() - start_time < VIDEO_LEN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            out.write(frame)
            cv.imshow("Gesture Recorder", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        count += 1
        print("saved")

cap.release()
cv.destroyAllWindows()