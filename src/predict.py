"""
For prediction on new images
"""

import numpy as np
from PIL import Image
from keras.models import load_model


emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

def predict_image(model_path, image_path):
    model = load_model(model_path)

    image = Image.open(image_path).convert("L")
    image = image.resize((48,48))

    image = np.array(image) / 255.0         #convert to np array and normalize

    image = np.expand_dims(image, axis = -1)
    image = np.expand_dims(image, axis = 0)

    prediction = model.predict(image)

    predicted_class = np.argmax(prediction)

    return emotion_map[predicted_class]