# Trains the model and exports the model

# Imports

import tensorflow as tf # Deep learning
import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import mediapipe as mp # Extracting keypoints
import os # File handling
import json # Json handling
from tqdm import tqdm
from model import Translator # Subclassed model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision

# Enable mixed precision globally
mixed_precision.set_global_policy('mixed_float16')
# Load data from .npy files
X_train = np.load("mediapipe_keypoints/X_train.npy")
y_train = np.load("mediapipe_keypoints/y_train.npy")
X_test = np.load("mediapipe_keypoints/X_test.npy")
y_test = np.load("mediapipe_keypoints/y_test.npy")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

asl_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'HELLO': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'NO': 15, 'O': 16, 'P': 17, 'Q': 18, 'R': 19,
    'S': 20, 'SORRY': 21, 'T': 22, 'THANKYOU': 23, 'U': 24, 'V': 25,
    'W': 26, 'X': 27, 'Y': 28, 'YES': 29, 'Z': 30, 'SPACE': 31
}

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
    save_best_only=True,     # only save when it's the best
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
    epochs=250,
    shuffle=True,
    batch_size=32,
    validation_data=[X_test, y_test],
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

print("\n\n\n")

print(model.evaluate(X_train, y_train))
print("\n\n\n")
print(model.evaluate(X_test, y_test))

print("\n\n\n")

model = load_model('best_model.keras')

print(model.evaluate(X_train, y_train))
print("\n\n\n")
print(model.evaluate(X_test, y_test))