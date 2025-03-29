from flask import Flask, request, jsonify, render_template, Response
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
import threading


app = Flask(__name__)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define model information
models_info = {
    "cnn_fe": {
        "file_id": "1Qs79c2AS8qB9UgzWEPEkFxSQnmvemWZE",
        "file_path": "models/cnn_fe.h5",
        "target_size": (256, 256),
        "preprocess": lambda x: x / 255.0,  # Rescale for CNN-FE
    },
    "en_finetuned": {
        "file_id": "14jqkfrtWbOceIEbghax2cfsWrlWrlSAh",
        "file_path": "models/efficientnetb7_transfer_learning.h5",
        "target_size": (600, 600),
        "preprocess": efficientnet_preprocess,  # EfficientNet preprocessing
    },
    "en_tl": {
        "file_id": "1TYyZup1KO4F7Zw4hYPX8EQF5KYl1IWzi",
        "file_path": "models/fine_tune_efficientnetb7_transfer_learning.h5",
        "target_size": (600, 600),
        "preprocess": efficientnet_preprocess,
    },
}

# Class labels
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

# Dictionary to store loaded models
loaded_models = {}
current_model = None
download_progress = {"status": "idle", "progress": 0, "model": None}
download_thread = None


def download_model_with_progress(model_choice):
    global download_progress, loaded_models

    # Reset progress
    download_progress = {"status": "downloading", "progress": 0, "model": model_choice}

    # Model info
    file_id = models_info[model_choice]["file_id"]
    file_path = models_info[model_choice]["file_path"]

    try:
        # Download the model - removed callback parameter
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            file_path,
            quiet=False,
        )

        # Update progress to 100% after download completes
        download_progress["progress"] = 100

        # Load the model
        loaded_models[model_choice] = load_model(file_path)
        download_progress = {
            "status": "complete",
            "progress": 100,
            "model": model_choice,
        }
    except Exception as e:
        download_progress = {
            "status": "error",
            "progress": 0,
            "model": model_choice,
            "error": str(e),
        }


def clean_up_models(except_model=None):
    """Clean up models except the specified one"""
    for model_choice, model_data in models_info.items():
        if model_choice != except_model and os.path.exists(model_data["file_path"]):
            try:
                # Remove from memory if loaded
                if model_choice in loaded_models:
                    del loaded_models[model_choice]

                # Remove file
                os.remove(model_data["file_path"])
            except Exception as e:
                print(f"Error deleting model {model_choice}: {e}")


@app.route("/download-model", methods=["POST"])
def download_model():
    global download_thread, current_model

    model_choice = request.json.get("model_choice")
    if model_choice not in models_info:
        return jsonify({"error": "Invalid model choice"}), 400

    # If already downloaded
    if model_choice in loaded_models:
        current_model = model_choice
        return jsonify({"status": "already_loaded", "model": model_choice})

    # If already downloading
    if (
        download_progress["status"] == "downloading"
        and download_progress["model"] == model_choice
    ):
        return jsonify(
            {"status": "downloading", "progress": download_progress["progress"]}
        )

    # Start download in a separate thread
    download_thread = threading.Thread(
        target=download_model_with_progress, args=(model_choice,), daemon=True
    )
    download_thread.start()
    current_model = model_choice

    # Clean up other models
    clean_up_models(except_model=model_choice)

    return jsonify({"status": "started", "model": model_choice})


@app.route("/download-progress", methods=["GET"])
def get_download_progress():
    return jsonify(download_progress)


@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model_choice", "cnn_fe")
    if model_choice not in models_info:
        return jsonify({"error": "Invalid model choice"}), 400

    # Check if model is downloaded and loaded
    if model_choice not in loaded_models:
        return (
            jsonify({"error": "Model not loaded. Please select the model first."}),
            400,
        )

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Get model info
    selected_model = loaded_models[model_choice]
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


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
