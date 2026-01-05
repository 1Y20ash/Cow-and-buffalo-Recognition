import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Flask App Setup
# ----------------------------
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Load Model (ONCE at startup)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cow_breed_model_gpu.weights.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model failed to load:", e)
    model = None

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print("❌ Labels file error:", e)
    labels = []

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "❌ Model not loaded on server", 500

    if "file" not in request.files:
        return "❌ No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "❌ No file selected", 400

    # Save uploaded image
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)

    try:
        # Image preprocessing
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        predicted_label = labels[predicted_index] if labels else "Unknown"

        return render_template(
            "result.html",
            prediction=predicted_label,
            confidence=f"{confidence:.2f}",
            image_path=img_path
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        return "❌ Error during prediction", 500

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
