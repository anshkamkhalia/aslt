# Sign Language Phrase Translator

This is project uses a live webcam feed and identifies one of 5 gestures being performed in sign language:

1. NO
2. YES
3. HELLO
4. SORRY
5. THANK YOU

To run: `pip install -r requirements.txt`

If you want to run the live camera, run: `python live_feed_phrase.py`

Main modules used:

- `tensorflow`
- `mediapipe`
- `numpy`
- `opencv-python`

To record your own data, simply change gesture settings in `gesture_recorder.py` and run `python gesture_recorder.py`. This will record 2 second video clips and save them into directories.

Model architecture (final):

- `TimeDistributed Dense (512, relu)`
- `TimeDistributed Dense (256, relu)`
- `TimeDistributed Dense (128, relu)`
- `LSTM (64, tanh)`
- `LSTM (32, tanh)`
- `Attention`
- `BatchNormalization`
- `Dense (5, softmax)`