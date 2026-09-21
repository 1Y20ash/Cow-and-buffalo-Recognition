# Copilot instructions for this workspace

## Project overview
This workspace contains a Flask web app for Indian cow/buffalo breed recognition. The app uploads an image, preprocesses it to $224 \times 224$, runs a TensorFlow/EfficientNet-based model, and shows the predicted breed and confidence.

## Key files
- [app.py](../app.py): Flask entry point. Main routes live here and the model is loaded at import time.
- [train_model_gpu.py](../train_model_gpu.py): Training script for the model. Keep its architecture aligned with [app.py](../app.py) if changing layers or preprocessing.
- [requirements.txt](../requirements.txt): Python dependencies for the app.
- [templates/index.html](../templates/index.html) and [templates/result.html](../templates/result.html): UI for upload and prediction results.
- [static/style.css](../static/style.css): Front-end styling.
- [models/cow_breed_model_gpu.weights.h5](../models/cow_breed_model_gpu.weights.h5) and [labels.txt](../labels.txt): pretrained weights and class labels used by the app.

## Development workflow
- Use the project virtual environment in [tf212](../tf212) when running Python locally on Windows.
- Install dependencies with `pip install -r requirements.txt` from the workspace root.
- Start the app with `python app.py` and open `http://127.0.0.1:5000`.
- The app expects uploaded files to be saved under [static/uploads](../static/uploads); that folder is created automatically.

## Project conventions
- Keep the model architecture in [train_model_gpu.py](../train_model_gpu.py) and [app.py](../app.py) consistent. If you change the input size, layers, preprocessing, or output head, update both files.
- Use Flask flash messages for user-facing errors in [app.py](../app.py) rather than silent failures.
- Preserve the existing upload flow: validate the file type, save it safely with `secure_filename`, and show a result page on success.
- Keep image preprocessing aligned with the training pipeline: resize to $224 \times 224$, convert to RGB, and use the EfficientNet preprocessing function.
- The training dataset directory is excluded from version control and is not part of the repo snapshot; avoid relying on it for app changes unless needed for retraining.

## Safe change guidelines
- Do not overwrite the pretrained weights or labels unless you mean to retrain or change the model output classes.
- Avoid making breaking changes to route names or template variable names without updating the corresponding templates.
- If changing the UI, keep the existing form structure and result page behavior intact.
- If you add new dependencies, update [requirements.txt](../requirements.txt) and keep the versions compatible with the existing TensorFlow stack.

## Validation tips
- There are no automated tests in this repo, so manual validation is the main check.
- After changing the Flask app, confirm that the home page loads and that a sample image produces a prediction without crashing.
- If TensorFlow import or GPU issues occur, prefer the bundled environment in [tf212](../tf212) before trying a global Python install.
