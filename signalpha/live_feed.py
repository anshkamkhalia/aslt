import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import Counter
from tensorflow.keras.models import load_model
from model import Translator, Attention  # your custom model
from tensorflow.keras import mixed_precision

# Enable mixed precision globally
mixed_precision.set_global_policy('mixed_float16')
# Load model
model = load_model("best_model.keras", custom_objects={"Translator": Translator, "Attention": Attention})

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ASL dictionary
asl_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'HELLO': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'NO': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19,
    'S': 20, 'SORRY': 21, 'T': 22, 'THANKYOU': 23, 'U': 24, 'V': 25,
    'W': 26, 'X': 27, 'Y': 28, 'YES': 29, 'Z': 30, 'SPACE': 31
}
idx_to_letter = {v:k for k,v in asl_dict.items()}

# Parameters
sequence_length = 50  # frames per sequence (same as training)
hold_frames_threshold = 15  # around half second at 30 fps

sequence_buffer = []
pred_buffer = []
chars = []

current_letter = None
hold_counter = 0

# Helper: pad frame keypoints to 2 hands (126 features)
def pad_frame(frame_points):
    FRAME_LENGTH = 21*3*2
    if len(frame_points) < FRAME_LENGTH:
        frame_points += [0.0] * (FRAME_LENGTH - len(frame_points))
    return frame_points[:FRAME_LENGTH]

# Start webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)  # mirror view
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frame_points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract keypoints
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            hand_points = []
            for lm in hand_landmarks.landmark:
                hand_points.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            frame_points.extend(hand_points)

    # Pad for missing hands
    frame_points = pad_frame(frame_points)
    sequence_buffer.append(frame_points)

    # Keep only last N frames
    if len(sequence_buffer) > sequence_length:
        sequence_buffer.pop(0)

    if len(sequence_buffer) == sequence_length:
        seq_array = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)
        pred_class_idx = np.argmax(model.predict(seq_array, verbose=0))
        predicted_letter = idx_to_letter[pred_class_idx]

        # Add to prediction buffer for smoothing
        pred_buffer.append(predicted_letter)
        if len(pred_buffer) > 5:
            pred_buffer.pop(0)

        # Determine most common letter in buffer
        most_common, count = Counter(pred_buffer).most_common(1)[0]

        # Only update hold counter if prediction is stable
        if most_common == current_letter:
            hold_counter += 1
        else:
            current_letter = most_common
            hold_counter = 1  # reset counter

        # Commit letter if held long enough
        if hold_counter >= hold_frames_threshold:
            if current_letter == "SPACE":
                chars.append(" ")
            else:
                chars.append(current_letter)
            pred_buffer = []
            hold_counter = 0  # reset after committing

        # Show predicted letter on frame
        cv.putText(frame, f"Prediction: {predicted_letter}", (10,50),
                cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
    
    cv.imshow("ASL Live Feed", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
result = "".join(chars)
print(f"\n\nPredicted ASL sequence:\n{result}\n\n")
