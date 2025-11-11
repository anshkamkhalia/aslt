# Tests the model using webcam

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from model_phrase import Translator, Attention
import mediapipe as mp
import time
from collections import deque, Counter
# Load model
model = load_model("best_model.keras", custom_objects={"Translator": Translator, "Attention": Attention})

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,      # VIDEO mode for speed
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ASL dictionary
# asl_dict = {
#     'R': 0,  'U': 1,  'I': 2,  'N': 3,  'G': 4,
#     'Z': 5,  'T': 6,  'S': 7,  'A': 8,  'F': 9,
#     'O': 10, 'H': 11, 'del': 12, 'nothing': 13, 'space': 14,
#     'M': 15, 'J': 16, 'C': 17, 'D': 18, 'V': 19,
#     'Q': 20, 'X': 21, 'E': 22, 'B': 23, 'K': 24,
#     'L': 25, 'Y': 26, 'P': 27, 'W': 28, 'YES': 29, 'NO': 30
# }

asl_dict = {
    'NO ': 0, 'YES ': 1, 'HELLO/GOOD_BYE ': 2, 'SORRY ': 3, "THANK_YOU ": 4,
    'HOW_ARE_YOU ': 5, 'I_AGREE ': 6
}
idx_to_letter = {v: k for k, v in asl_dict.items()}

chars = []

cap = cv.VideoCapture(0)

# initialize
last_commit_time = 0
hold_time = 1.0  # seconds to hold gesture
fps = cap.get(cv.CAP_PROP_FPS) or 30  # fallback if FPS fails
frames_to_hold = int(fps * hold_time)

sequence_buffer = deque(maxlen=60)  # LSTM expects 60 timesteps
pred_buffer = deque(maxlen=30)      # for stable letter detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Prepare keypoints for this frame
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            keypoints = np.array(
                sum([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in hand_landmarks.landmark], []),
                dtype=np.float32
            )

            sequence_buffer.append(keypoints)

            # Predict only if we have a full sequence
            if len(sequence_buffer) == 60:
                input_seq = np.array(sequence_buffer, dtype=np.float32).reshape(1, 60, 63)
                # Get the full softmax probabilities
                preds = model.predict(input_seq, verbose=0)[0]  # e.g., array([0.1, 0.2, 0.7])

                # Pick the class with the highest probability
                pred_class_idx = np.argmax(preds)
                predicted_letter = idx_to_letter[pred_class_idx]

                # Optional: ignore low-confidence predictions
                THRESHOLD = 0.7
                if preds[pred_class_idx] < THRESHOLD:
                    predicted_letter = "not recognized"

                # Add to prediction buffer for stability
                if predicted_letter != "not recognized":
                    pred_buffer.append(predicted_letter)
                else:
                    pass

                # Commit only if stable
                most_common, count = Counter(pred_buffer).most_common(1)[0]
                if count >= frames_to_hold:
                    if chars and chars[-1] == most_common:
                        pass
                    else:
                        if most_common == "space":
                            chars.append(" ")
                        elif most_common == "del":
                            if chars:
                                chars.pop()
                        else:
                            chars.append(most_common)
                        pred_buffer.clear()

            # Display predicted letter on screen
            display_letter = pred_buffer[-1] if pred_buffer else "-"
            cv.putText(frame, f"Prediction: {display_letter}", (10,50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

    cv.imshow("ASL Live Feed", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(chars)

result = "".join(chars)

print(f"\n\n{result}\n\n")

cap.release()
cv.destroyAllWindows()