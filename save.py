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

MAX_LEN = 75  # max frames per video
FEATURES = 63  # 21 keypoints × 3 coords

def apply_augmentations(keypoints):
    keypoints = np.array(keypoints)
    if np.random.rand() > 0.5:
        keypoints[::3] = 1 - keypoints[::3]  # Flip X

    keypoints += np.random.normal(0, 0.01, keypoints.shape)  # jitter

    scale = np.random.uniform(0.98, 1.02)
    offset_x = np.random.uniform(-0.01, 0.01)
    offset_y = np.random.uniform(-0.01, 0.01)

    keypoints[::3] = keypoints[::3] * scale + offset_x
    keypoints[1::3] = keypoints[1::3] * scale + offset_y
    return keypoints.tolist()

# Define paths for data
ROOT = "wlasl"
DATA = f"{ROOT}/videos"

# Debug print statement
def debug_print(txt):
    """A helper function to print text in a spaced-out way"""
    print(f"\n\n\n\n{txt}\n\n\n\n")

missing_files = [] # List containing the missing video files

# Load missing data
with open(f"{ROOT}/missing.txt", "r") as f:
    for line in f:
        missing_files.append(line.strip())

filenames = os.listdir(DATA) # Every filename contained in the training directory

# Load class list
class_map = {}

# Open file
with open(f"{ROOT}/wlasl_class_list.txt", "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)  # split only once
        if len(parts) == 2:
            idx, label = parts
            class_map[int(idx)] = label # Save entry

X_train, y_train, X_test, y_test = [], [], [], [] # Initialize as empty list

nslt_100, nslt_300, nslt_1000, nslt_2000 = None, None, None, None

# Load nslt_<n>.json files
with open(f"{ROOT}/nslt_100.json") as f:
    values = json.load(f)
    nslt_100 = values

with open(f"{ROOT}/nslt_300.json") as f:
    values = json.load(f)
    nslt_300 = values

with open(f"{ROOT}/nslt_1000.json") as f:
    values = json.load(f)
    nslt_1000 = values

with open(f"{ROOT}/nslt_2000.json") as f:
    values = json.load(f)
    nslt_2000 = values

# Collect all class occurrences
all_targets = []

json_files = [nslt_100, nslt_300, nslt_1000, nslt_2000]
for dataset in json_files:
    for video_id, entry in dataset.items():
        all_targets.append(entry["action"][0])

# Count frequencies
counter = Counter(all_targets)

# Pick top n most common classes

n = 2000 # Number of most common classes to use

top_n = [item[0] for item in counter.most_common(n)]
top_n_set = set(top_n)

# Build a class index mapping for these 500 classes
all_labels = set(all_targets)
class_index = {label: idx for idx, label in enumerate(sorted(all_labels))}

def process_video(filename: str) -> Tuple[List, List, List, List]:

    # Initialize mediapipe network
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,      # Set True for images, False for video
        max_num_hands=2,              # Detect up to 2 hands
        min_detection_confidence=0.5, # Minimum confidence to detect a hand
        min_tracking_confidence=0.5   # Minimum confidence for tracking landmarks
    )

    X_train_local, y_train_local, X_test_local, y_test_local = [], [], [], []

    if filename in missing_files:
        return X_train_local, y_train_local, X_test_local, y_test_local

    cap = cv.VideoCapture(f"{DATA}/{filename}")
    fps = cap.get(cv.CAP_PROP_FPS)
    skip = max(int(fps / 15), 1)
    frame_idx = 0
    prev_gray = None
    sequence = []

    video_num = filename.replace(".mp4", "")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        frame_resized = cv.resize(frame, (320, 320))
        gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

        if prev_gray is not None and np.mean(cv.absdiff(gray, prev_gray)) < 5:
            prev_gray = gray
            continue
        prev_gray = gray

        frame_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Lookup JSON for target
        target, subset = None, None
        for dataset in [nslt_100, nslt_300, nslt_1000, nslt_2000]:
            if video_num in dataset:
                entry = dataset[video_num]
                target = entry["action"][0]
                subset = entry["subset"]
                break

        if target is None or target not in top_n_set:
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_points = []
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    frame_points.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                sequence.append(frame_points)

    cap.release()

    # Pad/truncate
    seq_len = len(sequence)
    if seq_len < MAX_LEN:
        sequence.extend([np.zeros(FEATURES) for _ in range(MAX_LEN - seq_len)])
    else:
        sequence = sequence[:MAX_LEN]

    # Augmentations
    sequences = [
        sequence,
        [apply_augmentations(f) for f in sequence],
        [apply_augmentations(f) for f in sequence]
    ]
    target_idx = class_index[target]

    for seq in sequences:
        if subset == "train":
            X_train_local.append(seq)
            y_train_local.append(target_idx)
        else:
            X_test_local.append(seq)
            y_test_local.append(target_idx)

    
    hands.close()

    return X_train_local, y_train_local, X_test_local, y_test_local

X_train, y_train, X_test, y_test = [], [], [], []

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_video, f) for f in filenames]
    
    # Use tqdm directly over the list of futures
    for future in tqdm(futures, desc="Processing videos"):
        X_tr, y_tr, X_te, y_te = future.result()
        X_train.extend(X_tr)
        y_train.extend(y_tr)
        X_test.extend(X_te)
        y_test.extend(y_te)


cv.destroyAllWindows()

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