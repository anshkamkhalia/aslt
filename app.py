from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from collections import deque, Counter
from model_phrase import Translator, Attention
import os

app = Flask(__name__)

# Load model and MediaPipe once
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
asl_dict = {
    'NO ': 0, 'YES ': 1, 'HELLO ': 2, 'SORRY ': 3, "THANK_YOU ": 4
}
idx_to_letter = {v: k for k, v in asl_dict.items()}

sequence_buffer = deque(maxlen=60)  # LSTM expects 60 timesteps
pred_buffer = deque(maxlen=30)      # for stable letter detection

@app.route("/")
def home():
    return render_template("index.html")

chars = []

@app.route("/predict", methods=["GET", "POST"])
def predict_sign():

    # initialize
    last_commit_time = 0
    hold_time = 1.0  # seconds to hold gesture
    fps = 30  # fallback if FPS fails
    frames_to_hold = int(fps * hold_time)
    data = request.get_json()

    img_data = data["image"].split(",")[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

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

            return jsonify({"keypoints": keypoints})

if __name__ == "__main__":
    app.run(debug=True)