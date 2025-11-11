# Creates the model

import numpy as np # Mathematical operations and arrays
import cv2 as cv # Loading and reading images
import mediapipe as mp # Extracting keypoints
import os # File handling
import json # Json handling
from tqdm import tqdm # Progress bar
import tensorflow as tf # Deep learning

from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Attention, Layer, Conv2D, GlobalAveragePooling1D, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

# Enable mixed precision globally
mixed_precision.set_global_policy('mixed_float16')

# Attention layer
@register_keras_serializable(package="custom_layer")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation='tanh')
    
    def call(self, inputs):
        # inputs: (batch_size, timesteps, features)
        score = self.W(inputs)  # (batch_size, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)  # attention weights
        context = tf.reduce_sum(weights * inputs, axis=1)  # weighted sum
        return context
    
# Model
@register_keras_serializable(package="custom_model")
class Translator(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Sequence processing
        self.lstm1 = LSTM(64, return_sequences=True,
                          recurrent_regularizer=l2(1e-4),
                          kernel_regularizer=l2(1e-4))
        self.lstm2 = GRU(32, return_sequences=True,
                          recurrent_regularizer=l2(1e-4),
                          kernel_regularizer=l2(1e-4))
        self.attn = Attention()
        
        # Dense layers for classification
        self.dropout = Dropout(0.3)
        self.bn = BatchNormalization()
        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))
        self.dense2 = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))
        self.dense3 = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))
        self.out = Dense(32, activation="softmax", dtype='float32')  # 34 classes

    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.attn(x)
        
        x = self.dropout(x)
        x = self.bn(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return self.out(x)
