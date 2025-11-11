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

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Attention, Layer, Conv2D, GlobalAveragePooling1D, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import TimeDistributed

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
    def __init__(self, input_shape, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape_ = input_shape
        self.n_classes = n_classes
        # TimeDistributed Dense layers
        self.td_dense1 = TimeDistributed(Dense(512, activation='relu'))
        self.td_dense2 = TimeDistributed(Dense(256, activation='relu'))
        self.td_dense3 = TimeDistributed(Dense(128, activation='relu'))

        # LSTM layers
        self.lstm1 = LSTM(64, activation="tanh", return_sequences=True)
        self.lstm2 = LSTM(32, activation="tanh", return_sequences=True)
        
        # Attention layer
        self.attention = Attention()  # (batch, timesteps, features) -> (batch, features)
        
        # BatchNorm and Dropout applied after attention
        self.bn = BatchNormalization()  # normalize final vector
        self.dropout = Dropout(0.3)
        
        # Final output
        self.out = Dense(self.n_classes, activation='softmax')

    def call(self, x, training=False):
        # Ensure x is rank-3: (batch, timesteps, features)
        if len(x.shape) != 3:
            raise ValueError(f"Expected input of rank 3, got {x.shape}")
        x = self.td_dense1(x)
        x = self.td_dense2(x)
        x = self.td_dense3(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.attention(x)       # (batch, features)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)
        x = self.out(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_,
            "n_classes": self.n_classes
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)