"""Local production smoke test for the Render-style Flask application.

Run from the repository root after installing requirements.txt:
    PowerShell:
    $env:FLASK_SECRET_KEY="local-smoke-test-secret"
    py scripts/local_production_smoke_test.py

This intentionally uses Flask's test client rather than starting a second
server. The production entrypoint itself remains:
    gunicorn app:app
"""
import io
import os
import sys
from pathlib import Path

from PIL import Image

# Add the repository root to Python's import path when this script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("FLASK_SECRET_KEY", "local-smoke-test-secret")

try:
    from app import app
except Exception as exc:
    print(f"FAIL: application/model failed to load: {exc}")
    sys.exit(1)


def assert_status(response, expected, name):
    if response.status_code != expected:
        print(f"FAIL: {name}: expected {expected}, got {response.status_code}")
        return False
    print(f"PASS: {name}")
    return True


def main():
    app.config["TESTING"] = True
    client = app.test_client()
    ok = True

    # 1. Liveness and readiness
    ok &= assert_status(client.get("/health/live"), 200, "liveness endpoint")
    ready = client.get("/health/ready")
    ok &= assert_status(ready, 200, "readiness endpoint")
    if ready.status_code == 200:
        data = ready.get_json()
        if data.get("model") != "EfficientNetB0" or data.get("classes") != 42:
            print(f"FAIL: readiness payload is unexpected: {data}")
            ok = False
        else:
            print("PASS: readiness payload")

    # 2. Home page
    ok &= assert_status(client.get("/"), 200, "home page")

    # 3. Invalid extension must be rejected
    response = client.post(
        "/",
        data={"image": (io.BytesIO(b"not an image"), "sample.txt")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    ok &= assert_status(response, 302, "invalid extension rejection")

    # 4. Malformed image with an allowed extension must fail safely
    response = client.post(
        "/",
        data={"image": (io.BytesIO(b"not a real image"), "sample.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    ok &= assert_status(response, 302, "malformed image rejection")

    # 5. Real image must reach model inference and return the result page.
    image_buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(
        image_buffer, format="JPEG"
    )
    image_buffer.seek(0)

    response = client.post(
        "/",
        data={"image": (image_buffer, "smoke-test.jpg")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    ok &= assert_status(response, 200, "valid image prediction")
    if response.status_code == 200:
        body = response.get_data(as_text=True)
        for marker in ("Prediction Result", "Top 3 Model Predictions", "Confidence"):
            if marker not in body:
                print(f"FAIL: result page missing '{marker}'")
                ok = False
        if all(marker in body for marker in ("Prediction Result", "Top 3 Model Predictions", "Confidence")):
            print("PASS: prediction presentation")

    if ok:
        print("\nLOCAL PRODUCTION SMOKE TEST PASSED")
        return 0

    print("\nLOCAL PRODUCTION SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
