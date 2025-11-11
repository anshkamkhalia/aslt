from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from collections import deque, Counter
from model_phrase import Translator, Attention

app = Flask(__name__)

# Load model and MediaPipe once
model = load_model("best_model.keras", custom_objects={"Translator": Translator, "Attention": Attention})
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

asl_dict = {
    'NO ': 0, 'YES ': 1, 'HELLO ': 2, 'SORRY ': 3, "THANK_YOU ": 4
}
idx_to_letter = {v: k for k, v in asl_dict.items()}

# Buffers for stability
sequence_buffer = deque(maxlen=60)
pred_buffer = deque(maxlen=30)

@app.route("/", methods=["POST"])
def predict_sign():
    try:
        data = request.json
        frames = data.get("frames")  # list of base64-encoded frames
        if not frames:
            return jsonify({"error": "No frames received"}), 400

        for b64_frame in frames:
            img_bytes = base64.b64decode(b64_frame)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                base_x, base_y, base_z = (
                    hand_landmarks.landmark[0].x,
                    hand_landmarks.landmark[0].y,
                    hand_landmarks.landmark[0].z,
                )
                keypoints = np.array(
                    sum([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in hand_landmarks.landmark], []),
                    dtype=np.float32
                )
            else:
                keypoints = np.zeros(63, dtype=np.float32)

            sequence_buffer.append(keypoints)

        # Only predict if full sequence is ready
        if len(sequence_buffer) == 60:
            input_seq = np.array(sequence_buffer, dtype=np.float32).reshape(1, 60, 63)
            preds = model.predict(input_seq, verbose=0)[0]
            pred_idx = np.argmax(preds)
            confidence = float(np.max(preds))
            prediction = idx_to_letter[pred_idx]

            THRESHOLD = 0.7
            if confidence < THRESHOLD:
                prediction = "not recognized"

            # Stabilize predictions
            if prediction != "not recognized":
                pred_buffer.append(prediction)

            most_common, count = Counter(pred_buffer).most_common(1)[0]
            if count >= 15:  # stable for half of pred_buffer
                final_prediction = most_common
            else:
                final_prediction = "uncertain"

            return jsonify({
                "prediction": final_prediction,
                "confidence": confidence
            })

        else:
            return jsonify({"message": "Collecting frames..."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
