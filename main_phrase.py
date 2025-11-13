# Trains the model and exports the model

# Imports

import tensorflow as tf # Deep learning
import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import mediapipe as mp # Extracting keypoints
import os # File handling
import json # Json handling
from tqdm import tqdm
from model_phrase import Translator, Attention # Subclassed model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# Load data from .npy files
X_train = np.load("mediapipe_keypoints/X_train.npy")
y_train = np.load("mediapipe_keypoints/y_train.npy")
X_test = np.load("mediapipe_keypoints/X_test.npy")
y_test = np.load("mediapipe_keypoints/y_test.npy")
asl_dict = {
    'NO ': 0, 'YES ': 1, 'HELLO/GOOD_BYE ': 2, 'SORRY ': 3, "THANK_YOU ": 4,
    'HOW_ARE_YOU ': 5, 'I_AGREE ': 6, 'I_DISAGREE ': 7
}
# asl_dict = {
#     'R': 0,  'U': 1,  'I': 2,  'N': 3,  'G': 4,
#     'Z': 5,  'T': 6,  'S': 7,  'A': 8,  'F': 9,
#     'O': 10, 'H': 11, 'del': 12, 'nothing': 13, 'space': 14,
#     'M': 15, 'J': 16, 'C': 17, 'D': 18, 'V': 19,
#     'Q': 20, 'X': 21, 'E': 22, 'B': 23, 'K': 24,
#     'L': 25, 'Y': 26, 'P': 27, 'W': 28, 'YES': 29, 'NO': 30
# }
# Ensure shape: (num_samples, timesteps, features)
if len(X_train.shape) == 2:
    X_train = np.expand_dims(X_train, axis=1)  # (num_samples, 1, 63)
if len(X_test.shape) == 2:
    X_test = np.expand_dims(X_test, axis=1)   # (num_samples, 1, 63)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Initialize model
model = Translator()

# Create callbacks
# EarlyStopping: stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_accuracy',      # what to monitor
    patience=45,             # how many epochs to wait before stopping
    restore_best_weights=True
)

# ModelCheckpoint: save the best model during training
model_checkpoint = ModelCheckpoint(
    'best_model.keras',         # file to save
    monitor='val_accuracy',      # what to monitor
    save_best_only="val_accuracy",     # only save when it's the best
    verbose=1
)

# ReduceLROnPlateau: reduce learning rate if the model stops improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.75,              
    patience=10,              # wait this many epochs before reducing
    min_lr=1e-6,             # don't go below this LR
    verbose=1 
)

# Compile

model.compile(
    optimizer=Adam(0.005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=300,
    shuffle=True,
    batch_size=32,
    validation_data=[X_test, y_test],
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

print(model.evaluate(X_train, y_train))
print(model.evaluate(X_test, y_test))

model = load_model('best_model.keras')

print(model.evaluate(X_train, y_train))
print(model.evaluate(X_test, y_test))