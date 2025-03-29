from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess,
)
import numpy as np
import time
import os
from io import BytesIO
import gdown

os.makedirs("models", exist_ok=True)


file_id = "1Qs79c2AS8qB9UgzWEPEkFxSQnmvemWZE"
gdown.download(
    f"https://drive.google.com/uc?id={file_id}",
    "models/cnn_fe.h5",
    quiet=False,
)

file_id = "14jqkfrtWbOceIEbghax2cfsWrlWrlSAh"
gdown.download(
    f"https://drive.google.com/uc?id={file_id}",
    "models/efficientnetb7_transfer_learning.h5",
    quiet=False,
)

file_id = "1TYyZup1KO4F7Zw4hYPX8EQF5KYl1IWzi"
gdown.download(
    f"https://drive.google.com/uc?id={file_id}",
    "models/fine_tune_efficientnetb7_transfer_learning.h5",
    quiet=False,
)

app = Flask(__name__)

# Load models and define preprocessing steps
models_info = {
    "cnn_fe": {
        "model": load_model(r"models/cnn_fe.h5"),
        "target_size": (256, 256),
        "preprocess": lambda x: x / 255.0,  # Rescale for CNN-FE
    },
    "en_finetuned": {
        "model": load_model(r"models/fine_tune_efficientnetb7_transfer_learning.h5"),
        "target_size": (600, 600),
        "preprocess": efficientnet_preprocess,  # EfficientNet preprocessing
    },
    "en_tl": {
        "model": load_model(r"models/efficientnetb7_transfer_learning.h5"),
        "target_size": (600, 600),
        "preprocess": efficientnet_preprocess,
    },
}

# Class labels (assuming the same for all models)
class_names = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
]


@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model_choice", "cnn_fe")
    if model_choice not in models_info:
        return jsonify({"error": "Invalid model choice"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Retrieve model info
    selected_model = models_info[model_choice]["model"]
    target_size = models_info[model_choice]["target_size"]
    preprocess_func = models_info[model_choice]["preprocess"]

    try:
        img = load_img(BytesIO(file.read()), target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_func(img_array)  # Apply appropriate preprocessing
        img_array = np.expand_dims(img_array, axis=0)

        start_time = time.time()
        pred = selected_model.predict(img_array)
        elapsed_time = time.time() - start_time

        predicted_idx = np.argmax(pred, axis=1)[0]
        predicted_label = class_names[predicted_idx]

        return jsonify(
            {"predicted_class": predicted_label, "prediction_time": elapsed_time}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
