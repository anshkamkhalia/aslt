import cv2
import base64
import requests
import json
import mediapipe as mp

# API endpoint
API_URL = "https://aslt-api.onrender.com/predict"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a selfie view
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = hands.process(rgb_frame)

    # Check if hands are detected
    if result.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode frame as JPEG and send to API
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        payload = json.dumps({"image": img_base64})

        try:
            response = requests.post(API_URL, data=payload, headers={'Content-Type': 'application/json'})
            print("Prediction:", response.json())
        except Exception as e:
            print("Error:", e)
    else:
        cv2.putText(frame, "No hands detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display webcam feed
    cv2.imshow("Webcam", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()