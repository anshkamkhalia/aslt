# Trains the model and exports the model

# Imports

import tensorflow as tf # Deep learning
import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import mediapipe as mp # Extracting keypoints
import os # File handling
import json # Json handling
from tqdm import tqdm
from model import Translator, Attention # Subclassed model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# Load data from .npy files
X_train = np.load("mediapipe_keypoints/X_train.npy")
y_train = np.load("mediapipe_keypoints/y_train.npy")
X_test = np.load("mediapipe_keypoints/X_test.npy")
y_test = np.load("mediapipe_keypoints/y_test.npy")

# Initialize model
model = Translator(input_shape=(32,63), n_classes=2000)

# Create callbacks
# EarlyStopping: stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',      # what to monitor
    patience=45,             # how many epochs to wait before stopping
    restore_best_weights=True
)

# ModelCheckpoint: save the best model during training
model_checkpoint = ModelCheckpoint(
    'wlasl_best.keras',         # file to save
    monitor='val_loss',      # what to monitor
    save_best_only=True,     # only save when it's the best
    verbose=1
)

# ReduceLROnPlateau: reduce learning rate if the model stops improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.75,              
    patience=10,              # wait this many epochs before reducing
    min_lr=1e-6,             # don't go below this LR
    verbose=1 
)

# Compile
model.compile(
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0),
    loss = 'sparse_categorical_crossentropy',
    metrics=["accuracy"]
)
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)


model.fit(
    X_train,
    y_train,
    epochs=500,
    shuffle=True,
    batch_size=128,
    validation_data=[X_test, y_test],
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

model = load_model('wlasl_best.keras', custom_objects={'Translator': Translator,
                                                       'Attention': Attention})

print(f"\n\n\n{model.evaluate(X_train, y_train)}")
print(f"\n\n\n{model.evaluate(X_test, y_test)}")
