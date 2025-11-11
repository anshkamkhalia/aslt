from tensorflow import keras
from model_phrase import Attention, Translator

# Load your .keras model
model = keras.models.load_model("best_model.keras", custom_objects={
    'Translator': Translator,
    'Attention': Attention
})

# Save it as .h5
model.save("model.h5")