# Creates and saves datasets and labels containing keypoints

# Imports

import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error
# Suppress MediaPipe logs
os.environ['GLOG_minloglevel'] = '3'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
import mediapipe as mp # Extracting keypoints
import json # Json handling
from tqdm import tqdm # Progress bar
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallelization
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

# Enable mixed precision globally
mixed_precision.set_global_policy('mixed_float16')
# Data preprocessing pipeline:
# 1. Load missing filenames
# 2. Load video files and exclude missing files
# 3. Read image data
# 4. Resize/normalize images
# 5. Apply augmentations
# 6. Extract MediaPipe keypoints
# 7. Save to X_train or X_test
# 8. Search in JSON files for targets
# 9. Save labels to y_train or y_test
# Repeat for all files

NUM_HANDS = 2
NUM_LANDMARKS = 21
NUM_COORDS = 3
FRAME_LENGTH = NUM_HANDS * NUM_LANDMARKS * NUM_COORDS  # 126

def pad_frame(frame_points):
    """Pad frame to 2 hands worth of keypoints (126)"""
    if len(frame_points) < FRAME_LENGTH:
        frame_points += [0.0] * (FRAME_LENGTH - len(frame_points))
    return frame_points[:FRAME_LENGTH]  # truncate if too long


asl_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'HELLO': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'NO': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19,
    'S': 20, 'SORRY': 21, 'T': 22, 'THANKYOU': 23, 'U': 24, 'V': 25,
    'W': 26, 'X': 27, 'Y': 28, 'YES': 29, 'Z': 30, 'SPACE': 31
}

# Strengthened augmentations

def apply_augmentations(keypoints):
    keypoints = np.array(keypoints, dtype=np.float32)
    
    # Horizontal flip
    if np.random.rand() > 0.5:
        keypoints[::3] = 1 - keypoints[::3]

    # Jitter, scaling, offsets
    keypoints += np.random.normal(0, 0.15, keypoints.shape)
    scale = np.random.uniform(0.9, 1.1)
    keypoints[::3] *= scale
    keypoints[1::3] *= scale
    offset_x = np.random.uniform(-0.08, 0.08)
    offset_y = np.random.uniform(-0.08, 0.08)
    keypoints[::3] += offset_x
    keypoints[1::3] += offset_y

    # Rotation
    angle = np.random.uniform(-15, 15) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = keypoints[::3].copy()
    y = keypoints[1::3].copy()
    keypoints[::3] = cos_a * x - sin_a * y
    keypoints[1::3] = sin_a * x + cos_a * y

    # ENSURE shape
    if len(keypoints) < FRAME_LENGTH:
        keypoints = np.pad(keypoints, (0, FRAME_LENGTH - len(keypoints)), 'constant')
    elif len(keypoints) > FRAME_LENGTH:
        keypoints = keypoints[:FRAME_LENGTH]

    return keypoints.tolist()


# Data paths
ROOT = "SignAlphaSet/ASL_dynamic"

X_train = []
y_train = []

def process_video(filename: str, character: str, target: int) -> Tuple[List, List]:
    
    # Initialize mediapipe network
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    X_train_local, y_train_local = [], []

    cap = cv.VideoCapture(f"{ROOT}/{character}/{filename}")
    sequence = []
    prev_gray = None  # initialize for motion detection
    SEQUENCE_LENGTH = 50  # desired frames per sequence

    # Params for frame skipping -> increases speed
    skip_rate = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count+=1
        if frame_count % skip_rate != 0:
            continue # skips this frame

        frame_resized = cv.resize(frame, (224, 224))
        gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

        # Skip nearly identical frames
        if prev_gray is not None and np.mean(cv.absdiff(gray, prev_gray)) < 5:
            prev_gray = gray
            continue
        prev_gray = gray

        frame_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Extract keypoints for up to 2 hands
        frame_points = []
        for hand_idx in range(2):
            if results.multi_hand_landmarks and hand_idx < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[hand_idx]
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    frame_points.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            else:
                frame_points.extend([0.0] * (21 * 3))  # pad missing hand

        frame_points = pad_frame(frame_points=frame_points)
        sequence.append(frame_points)

    cap.release()
    hands.close()

    # Ensure sequence has fixed length
    if len(sequence) >= SEQUENCE_LENGTH:
        sequence = sequence[:SEQUENCE_LENGTH]
    else:
        # pad with zeros
        padding = [[0.0]*126] * (SEQUENCE_LENGTH - len(sequence))
        sequence += padding

    # Augmentations
    sequences = [sequence] + [[apply_augmentations(f) for f in sequence] for _ in range(7)]
    for seq in sequences:
        X_train_local.append(seq)
        y_train_local.append(target)

    return X_train_local, y_train_local


directories = os.listdir(ROOT)

for dir in directories:

    if dir == ".DS_Store":
        continue

    cwd = f"{ROOT}/{dir}" # change current directory
    files = os.listdir(cwd)  # list files

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_video, filename=f, character=dir, target=asl_dict[dir]) for f in files]

        for future in tqdm(futures, desc="processing"):
            X_tr, y_tr = future.result()
            X_train.extend(X_tr)
            y_train.extend(y_tr)

# Add spaces manually as original dataset does not come with them
# Taking spaces from a different dataset

SPACE_DIR = "SignAlphaSet/space"
space_images = os.listdir(SPACE_DIR)

# Assuming the next available target index for 'space' is 31 (update if different)
space_target = 31

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # single images
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for img_file in tqdm(space_images, desc="Processing SPACE images"):
    img_path = os.path.join(SPACE_DIR, img_file)
    img = cv.imread(img_path)
    if img is None:
        continue  # skip corrupted/missing files

    img_resized = cv.resize(img, (224, 224))
    img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            frame_points = []
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            for lm in hand_landmarks.landmark:
                frame_points.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
            # Replicate single image into a sequence of 30 frames
            frame_points = pad_frame(frame_points=frame_points)
            sequence = [frame_points] * 100
            
            # Create augmented versions
            sequences = [sequence] + [[apply_augmentations(f) for f in sequence] for _ in range(6)]
            
            for seq in sequences:
                X_train.append(seq)
                y_train.append(space_target)

hands.close()

cv.destroyAllWindows()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=42)

X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

indices = np.arange(len(X_train))
np.random.shuffle(indices) # shuffles

X_train = X_train[indices]
y_train = y_train[indices]

indices = np.arange(len(X_test))
np.random.shuffle(indices) # shuffles

X_test = X_test[indices]
y_test = y_test[indices]

# Save to .npy files
np.save("mediapipe_keypoints/X_train.npy", X_train)
np.save("mediapipe_keypoints/y_train.npy", y_train)
np.save("mediapipe_keypoints/X_test.npy", X_test)
np.save("mediapipe_keypoints/y_test.npy", y_test)



