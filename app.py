from flask import Flask, render_template, request, redirect, flash
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# Configuration
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "cow_breed_model_gpu.weights.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

secret_key = os.environ.get("FLASK_SECRET_KEY")
if not secret_key:
    raise RuntimeError("FLASK_SECRET_KEY environment variable is required.")
app.secret_key = secret_key

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
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32"
    )(x)
    return tf.keras.Model(inputs=inp, outputs=out)

# ==========================
# Load labels
# ==========================
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    if len(labels) != 42:
        raise ValueError(f"Expected exactly 42 labels, found {len(labels)}.")
    if len(set(labels)) != len(labels):
        raise ValueError("labels.txt contains duplicate labels.")

    idx_to_class = {i: name for i, name in enumerate(labels)}
    print(f"Loaded and validated {len(labels)} labels")
except Exception as e:
    raise RuntimeError(f"Label loading/validation failed: {e}") from e

# ==========================
# Load model weights
# ==========================
def load_production_model():
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    production_model = build_model(len(labels))
    output_shape = production_model.output_shape

    if len(output_shape) != 2 or output_shape[-1] != len(labels):
        raise RuntimeError(
            "Model output/label mismatch: "
            f"model outputs {output_shape[-1] if output_shape else 'unknown'} "
            f"classes, but labels.txt contains {len(labels)}."
        )

    try:
        production_model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights from {WEIGHTS_PATH}: {e}"
        ) from e

    print(
        "Production model loaded successfully: "
        f"EfficientNetB0 → {output_shape[-1]} classes"
    )
    return production_model

model = load_production_model()

# ==========================
# Helpers
# ==========================
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(file_stream):
    """Decode and preprocess an uploaded image entirely in memory."""
    with Image.open(file_stream) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).astype("float32")

    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ==========================
# Health & Readiness
# ==========================
@app.route("/health/live", methods=["GET"])
def health_live():
    """Liveness probe: confirms the Flask process is responding."""
    return {"status": "ok"}, 200


@app.route("/health/ready", methods=["GET"])
def health_ready():
    """Readiness probe: confirms the model and labels are loaded."""
    if model is None or len(labels) != 42:
        return {
            "status": "not_ready",
            "reason": "production model or labels are unavailable"
        }, 503

    return {
        "status": "ready",
        "model": "EfficientNetB0",
        "classes": len(labels)
    }, 200

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
            if model is None:
                flash("Model not loaded on server.", "danger")
                return redirect(request.url)

            try:
                x = prepare_image(file.stream)
                preds = model.predict(x, verbose=0)[0]

                top_indices = np.argsort(preds)[-3:][::-1]
                predictions = [
                    {
                        "breed": idx_to_class.get(int(index), "Unknown"),
                        "confidence": round(float(preds[index]) * 100, 2)
                    }
                    for index in top_indices
                ]

                top_prediction = predictions[0]
            except Exception as e:
                flash(f"Prediction error: {e}", "danger")
                return redirect(request.url)

            return render_template(
                "result.html",
                filename=file.filename,
                breed=top_prediction["breed"],
                confidence=top_prediction["confidence"],
                predictions=predictions
            )

        flash("Allowed file types: png, jpg, jpeg", "danger")
        return redirect(request.url)

    return render_template("index.html")

# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
