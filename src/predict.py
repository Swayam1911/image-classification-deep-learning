import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model("model/flower_model.h5")


with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

def predict(img):
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = float(100 * np.max(score))

    return predicted_class, confidence

