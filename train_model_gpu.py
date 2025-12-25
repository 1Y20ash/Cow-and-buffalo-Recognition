# train_model_gpu.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # reduce TF logging
os.environ["KMP_AFFINITY"] = "none"

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# -------- HARD DISABLE TENSORBOARD ----------
from tensorflow.keras.callbacks import TensorBoard

class DisabledTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

# override TensorBoard everywhere
callbacks.TensorBoard = DisabledTensorBoard
# --------------------------------------------

# --------------------------------------------------
#  Ensure eager functions -> avoids some serialization issues while we compute class weights
# --------------------------------------------------
tf.config.run_functions_eagerly(True)

# Suppress TF logger messages
tf.get_logger().setLevel("ERROR")

# --------------------------------------------------
# Robust NoLogging callback (single, safe implementation)
# Converts tensors, arrays, lists -> single floats (or None) so JSON serialization can't fail.
# --------------------------------------------------
class NoLogging(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        safe = {}
        for k, v in logs.items():
            # Unwrap TF EagerTensors
            try:
                if hasattr(v, "numpy"):
                    v = v.numpy()
            except Exception:
                pass

            # Numpy arrays -> Python lists
            try:
                if hasattr(v, "tolist"):
                    v = v.tolist()
            except Exception:
                pass

            # If it's list/tuple/ndarray, reduce to a single float (mean)
            if isinstance(v, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(v, dtype=np.float64)
                    if arr.size == 1:
                        v = float(arr.item())
                    else:
                        v = float(np.mean(arr))
                except Exception:
                    v = None

            # Try final float conversion
            try:
                v = float(v)
            except Exception:
                v = None

            safe[k] = v

        # Replace logs contents so Keras/TensorFlow won't try to JSON-serialize complex objects
        logs.clear()
        logs.update(safe)

        # Print a compact summary for visibility (optional)
        print(f"Epoch {epoch+1} logs: {safe}")

nolog = NoLogging()

# --------------------------------------------------
#  GPU check + mixed precision
# --------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
print("GPUs found:", gpus)

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# Use mixed precision for speed on supported GPUs
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed Precision Enabled")
except Exception as e:
    print("Mixed precision not enabled:", e)

# --------------------------------------------------
#  Dataset detection (auto-find the dataset folder)
# --------------------------------------------------
possible_paths = [
    r"C:\Users\yashc\OneDrive\Desktop\sih_updated\Indian_bovine_breeds\Indian_bovine_breeds",
    r"C:\Users\yashc\OneDrive\Desktop\sih_updated\Indian_bovine_breeds",
    os.path.join(os.getcwd(), "Indian_bovine_breeds"),
    os.path.join(os.getcwd())
]

DATA_DIR = None
for p in possible_paths:
    if p and os.path.isdir(p):
        try:
            sub = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(sub) >= 2:
                DATA_DIR = p
                break
        except Exception:
            pass

if DATA_DIR is None:
    for root, dirs, files in os.walk(os.getcwd()):
        rel = os.path.relpath(root, os.getcwd())
        if rel.count(os.sep) > 4:
            continue
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) >= 3:
            DATA_DIR = root
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory automatically. "
        "Place the dataset folder (containing subfolders for each class) inside the current working directory "
        "or set DATA_DIR manually in the script."
    )

print("Using dataset directory:", DATA_DIR)

# --------------------------------------------------
#  Dataset loading (safe parameters)
# --------------------------------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42
VAL_SPLIT = 0.2

train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Detected classes:", num_classes)

# performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# --------------------------------------------------
#  Class weights (robust)
# --------------------------------------------------
def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for images, labels in dataset:
        try:
            lbls = labels.numpy()
        except Exception:
            lbls = list(labels)
        for lbl in lbls:
            counts[int(lbl)] += 1.0

    counts = np.where(counts == 0, 1.0, counts)
    total = counts.sum()
    class_weights = {i: float(total / (num_classes * counts[i])) for i in range(num_classes)}
    print("Class Weights:", class_weights)
    return class_weights

class_weights = compute_class_weights(train_ds, num_classes)

# --------------------------------------------------
#  Build model (EfficientNetB0 as backbone)
# --------------------------------------------------
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
# final layer must output float32 to avoid mixed-precision dtype issues in loss
out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

model = models.Model(inputs=inp, outputs=out)
model.summary()

# --------------------------------------------------
#  Compile
# --------------------------------------------------
optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
#  Callbacks (place AFTER compile)
# --------------------------------------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_model.weights.h5",
    save_best_only=True,
    monitor="val_loss",
    save_weights_only=True,
    save_format="h5"
)


earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# safe callbacks only: earlystop + checkpoint + no logging
# safe callbacks only: run NoLogging first so it sanitizes logs for all other callbacks
callback_list = [nolog, earlystop_cb, checkpoint_cb]


# --------------------------------------------------
#  Prepare for training (performance preferences)
# --------------------------------------------------
# Turn eager functions off for better GPU performance during training
tf.config.run_functions_eagerly(False)
model.run_eagerly = False

# --------------------------------------------------
#  Train
# --------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=callback_list
)

# Save labels
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("Labels saved:", len(class_names))
# Save labels
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("Labels saved:", len(class_names))


# --------------------------------------------------
#  Save final model
# --------------------------------------------------
model.save_weights("cow_breed_model_gpu.weights.h5")
print("Weights saved successfully!")


