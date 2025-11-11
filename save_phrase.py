# Creates and saves datasets and labels containing keypoints

# Imports

import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import os # File handling
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
# Suppress MediaPipe logs
os.environ['GLOG_minloglevel'] = '3'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
import mediapipe as mp # Extracting keypoints
import json # Json handling
from tqdm import tqdm # Progress bar
from sklearn.model_selection import train_test_split # Splitting data
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallelization
from typing import Tuple, List # Function dtype annotation

# Establish directories
ROOT = "archive"
TRAIN = f"{ROOT}/asl_alphabet_train/asl_alphabet_train"
TEST = f"{ROOT}/asl_alphabet_test/asl_alphabet_test"

# Get list of dir names in training folder
dirs = os.listdir(TRAIN)

X_train, y_train = [], []

# Targets
asl_dict = {
    'NO ': 0, 'YES ': 1, 'HELLO/GOOD_BYE ': 2, 'SORRY ': 3, "THANK_YOU ": 4,
    'HOW_ARE_YOU ': 5, 'I_AGREE ': 6
}

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

TARGET_FRAMES = 60  # match videos

def process_image(dir: str, filename: str) -> Tuple[List, List]:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, 
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    X_local, y_local = [], []

    img = cv.imread(f"{TRAIN}/{dir}/{filename}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = hands.process(img)
    sequence = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            frame_points = []
            for lm in hand_landmarks.landmark:
                frame_points.extend([lm.x, lm.y, lm.z])
            sequence.append(frame_points)

    # If no hands detected, fill with zeros
    if not sequence:
        sequence = [[0.0]*63]

    # If only one hand detected, duplicate to mimic LSTM input length
    while len(sequence) < TARGET_FRAMES:
        sequence.append(sequence[-1])

    # Only keep TARGET_FRAMES
    sequence = sequence[:TARGET_FRAMES]

    # Augment
    seq_aug = [apply_augmentations(f) for f in sequence]

    X_local.append(seq_aug)
    y_local.append(asl_dict[dir])

    hands.close()
    return X_local, y_local


# Create a helper to submit jobs
def submit_videos(file_list, directory, target_idx, repeats=1):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for f in file_list:
            if f == ".DS_Store":
                continue
            for _ in range(repeats):
                futures.append(executor.submit(process_video_file, f"{directory}/{f}", target_idx))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {directory.split('/')[-1]}"):
            seq_aug, t_idx = future.result()
            results.append((seq_aug, t_idx))
    return results

# Process gestures
DIRECTORY = "data"
YES = f"{DIRECTORY}/YES"
NO = f"{DIRECTORY}/NO"
HELLO = f"{DIRECTORY}/HELLO"
SORRY = f"{DIRECTORY}/SORRY"
THANK_YOU = f"{DIRECTORY}/THANK_YOU"
HOW_ARE_YOU = f"{DIRECTORY}/HOW_ARE_YOU"
I_AGREE = f"{DIRECTORY}/I_AGREE"

# List of files
yes_files = os.listdir(YES)
no_files = os.listdir(NO)
hello_files = os.listdir(HELLO)
sorry_files = os.listdir(SORRY)
thank_you_files = os.listdir(THANK_YOU)
how_are_you_files = os.listdir(HOW_ARE_YOU)
i_agree_files = os.listdir(I_AGREE)

# Initialize mediapipe network
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # Set True for images, False for video
    max_num_hands=2,              # Detect up to 2 hands
    min_detection_confidence=0.5, # Minimum confidence to detect a hand
    min_tracking_confidence=0.5   # Minimum confidence for tracking landmarks
)

TARGET_FRAMES = 60  # we want exactly 60 frames per sequence

def process_video_file(file_path, target_idx):
    cap = cv.VideoCapture(file_path)
    frames = []

    while len(frames) < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1])
            else:
                break
            continue
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    frames = frames[:TARGET_FRAMES]

    keypoint_sequence = []
    for frame in frames:
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_points = []
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    frame_points.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                keypoint_sequence.append(frame_points)
        else:
            keypoint_sequence.append([0.0] * 63)

    if len(keypoint_sequence) == 0:
        keypoint_sequence = [[0.0] * 63 for _ in range(TARGET_FRAMES)]

    seq_aug = [apply_augmentations(f) for f in keypoint_sequence]

    return seq_aug, target_idx  # return instead of appending globally

# Collect all processed data safely - changing global variables is not thread-safe
all_data = []

all_data.extend(submit_videos(yes_files, YES, target_idx=1, repeats=3))
all_data.extend(submit_videos(no_files, NO, target_idx=0, repeats=3))
all_data.extend(submit_videos(hello_files, HELLO, target_idx=2, repeats=4))
all_data.extend(submit_videos(sorry_files, SORRY, target_idx=3, repeats=4))
all_data.extend(submit_videos(thank_you_files, THANK_YOU, target_idx=4, repeats=4))
all_data.extend(submit_videos(how_are_you_files, HOW_ARE_YOU, target_idx=5, repeats=4))
all_data.extend(submit_videos(i_agree_files, I_AGREE, target_idx=6, repeats=4))

# Unpack into X_train and y_train
X_train = [seq for x, y in all_data for seq in x]  # flatten sequences
y_train = [y for x, y in all_data for _ in x]      # repeat labels
# Split data into test and train
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

cv.destroyAllWindows()
hands.close()
for i, x in enumerate(X_train):
    print(i, np.array(x).shape)

for i, x in enumerate(X_test):
    print(i, np.array(x).shape)

for i, x in enumerate(y_train):
    print(i, np.array(x).shape)

for i, x in enumerate(y_test):
    print(i, np.array(x).shape)

X_train = np.array(X_train, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.uint8)
y_test  = np.array(y_test, dtype=np.uint8)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Save training and testing sets
np.save("mediapipe_keypoints/X_train.npy", np.array(X_train))
np.save("mediapipe_keypoints/y_train.npy", np.array(y_train))
np.save("mediapipe_keypoints/X_test.npy", np.array(X_test))
np.save("mediapipe_keypoints/y_test.npy", np.array(y_test))