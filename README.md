# ASL Phrase Translator

Real-time sign-language phrase recognition using a webcam, MediaPipe hand landmarks, and a sequence model (LSTM + attention) built with TensorFlow/Keras.

This repository includes:
- A phrase-level pipeline (current focus): record gestures, extract keypoints, train model, run live prediction, serve predictions with Flask.
- Legacy experiments for alphabet/WLASL classification (kept in the repo but separate from the phrase workflow).

## Supported Phrase Classes

The phrase model currently predicts these 8 classes:

1. `NO`
2. `YES`
3. `HELLO/GOOD_BYE`
4. `SORRY`
5. `THANK_YOU`
6. `HOW_ARE_YOU`
7. `I_AGREE`
8. `I_DISAGREE`

## How the Phrase Pipeline Works

1. Record short videos per phrase (`data/<PHRASE>/*.mp4`).
2. Convert each video into a fixed-length sequence of hand keypoints with MediaPipe.
3. Apply light augmentation (flip/jitter/scale/offset) on keypoints.
4. Train a sequence classifier on saved `.npy` tensors.
5. Run real-time inference with temporal smoothing to stabilize predictions.

Input representation:
- 60 timesteps per sample.
- 63 features per timestep (21 hand landmarks x `[x, y, z]`).
- Landmarks are made relative to wrist coordinates (`landmark 0`) for translation invariance.

## Model Architecture (Phrase)

Defined in `model_phrase.py`:

1. `TimeDistributed(Dense(512, relu))`
2. `TimeDistributed(Dense(256, relu))`
3. `TimeDistributed(Dense(128, relu))`
4. `LSTM(64, return_sequences=True)`
5. `LSTM(32, return_sequences=True)`
6. Custom attention pooling layer
7. `BatchNormalization`
8. `Dense(8, softmax)`

Training script: `main_phrase.py`  
Saved model artifact: `best_model.keras`

## Repository Map (Important Files)

- `gesture_recorder.py`: capture labeled phrase clips from webcam into `data/<GESTURE>/`.
- `save_phrase.py`: preprocess phrase videos into `mediapipe_keypoints/X_*.npy` and `y_*.npy`.
- `model_phrase.py`: custom `Translator` model and custom attention layer.
- `main_phrase.py`: phrase model training and evaluation.
- `live_feed_phrase.py`: local webcam inference with on-screen prediction.
- `app.py`: Flask API endpoint (`/predict`) for frame-based inference.
- `requirements.txt`: Python dependencies.
- `runtime.txt`, `ProcFile`: deployment/runtime hints (Python and process command).

Legacy modules:
- `save.py`, `model.py`, `main.py`, and `signalpha/` are earlier/alternate pipelines.

## Setup

Python version in repo: `3.10.15` (`runtime.txt`).

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## End-to-End Usage (Phrase Model)

### 1) Record Data

Edit `GESTURE` in `gesture_recorder.py`, then run:

```bash
python gesture_recorder.py
```

Controls:
- Press `r` to record a 2-second clip.
- Press `q` to quit.

Repeat for each phrase class.

### 2) Build Training Tensors

```bash
python save_phrase.py
```

This creates:
- `mediapipe_keypoints/X_train.npy`
- `mediapipe_keypoints/y_train.npy`
- `mediapipe_keypoints/X_test.npy`
- `mediapipe_keypoints/y_test.npy`

### 3) Train

```bash
python main_phrase.py
```

Training uses:
- `Adam(0.005)`
- `sparse_categorical_crossentropy`
- callbacks: early stopping, LR reduction, model checkpointing

### 4) Run Local Live Inference

```bash
python live_feed_phrase.py
```

- Webcam feed opens with hand landmarks and current prediction.
- Press `q` to quit.

### 5) Run API Server (Optional)

```bash
python app.py
```

`POST /predict` expects JSON with base64 frames:

```json
{
  "frames": ["<base64_jpeg_frame_1>", "<base64_jpeg_frame_2>"]
}
```

Typical response:

```json
{
  "prediction": "YES ",
  "confidence": 0.93
}
```

## Notes and Current Limitations

- `live_feed_phrase.py` and `app.py` use confidence thresholding and majority smoothing for stability.
- Data quality and class balance strongly affect performance.
- The phrase list in code uses trailing spaces in class names (for compatibility with existing mappings).
- `templates/index.html` is currently a rough prototype and may not match `/predict` payload shape without updates.

## Troubleshooting

- Camera not opening: check webcam permissions and `CAM_INDEX` in recorder/inference scripts.
- Empty or poor predictions: collect more varied clips, retrain, and verify lighting/background.
- Model load errors: ensure custom objects are passed when loading (`Translator`, `Attention`), as done in code.
