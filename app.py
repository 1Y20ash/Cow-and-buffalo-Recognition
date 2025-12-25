from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# Configuration
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "cow_breed_model_gpu.weights.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "a-very-secret-key"

# ==========================
# Rebuild SAME architecture as training script
# ==========================
def build_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inp = tf.keras.layers.Input(shape=(224, 224, 3))
    x = preprocess_input(inp)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # final float32 layer (same as training)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

# ==========================
# Load labels
# ==========================
try:
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    idx_to_class = {i: name for i, name in enumerate(labels)}
    print(f"✅ Loaded {len(labels)} labels")
except Exception as e:
    labels = []
    idx_to_class = {}
    print(f"❌ Failed to load labels: {e}")

# ==========================
# Load model weights
# ==========================
model = None
try:
    model = build_model(len(idx_to_class))
    model.load_weights(WEIGHTS_PATH)
    print("✅ Loaded model weights successfully!")
except Exception as e:
    print(f"❌ Failed to load model weights: {e}")
    model = None

# ==========================
# Helpers
# ==========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            if model is None:
                flash("Model not loaded on server.", "danger")
                return redirect(request.url)

            try:
                x = prepare_image(save_path)
                preds = model.predict(x)[0]
                top_idx = int(np.argmax(preds))
                confidence = float(preds[top_idx])
                breed = idx_to_class.get(top_idx, "Unknown")
            except Exception as e:
                flash(f"Prediction error: {e}", "danger")
                return redirect(request.url)

            return render_template(
                "result.html",
                filename=filename,
                breed=breed,
                confidence=round(confidence * 100, 2)
            )

        else:
            flash("Allowed file types: png, jpg, jpeg", "danger")
            return redirect(request.url)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))

# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

