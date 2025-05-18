from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.h5")  # Replace with your model path

# You can set class names manually if needed
class_names = ["Class A", "Class B", "Class C"]

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_file = request.files['file']
    img = image.load_img(img_file, target_size=(224, 224))  # Adjust size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "class": predicted_class,
        "confidence": confidence
    })
