"""Flask routes for face recognition."""

from flask import Blueprint, request, jsonify
import os
from dotenv import load_dotenv
import psutil

load_dotenv()

from utils.face_detector import detect_and_crop_face
from utils.model_loader import load_model, predict_identity

bp = Blueprint("main", __name__)
process = psutil.Process()

# Global variable to store the loaded model
_model = None
_model_path = None


def get_model():
    """Load and cache the TensorFlow model."""
    global _model, _model_path
    model_zip_path = os.environ.get("MODEL_PATH", "models/face_model.zip")

    if _model is None or _model_path != model_zip_path:
        _model = load_model(model_zip_path)
        _model_path = model_zip_path

    return _model


@bp.route("/recognize", methods=["POST"])
def recognize():
    """
    Endpoint to recognize a face from an uploaded image.

    Expects:
        - A .jpg image file in the request

    Returns:
        JSON response with:
        - recognized: bool (whether a face was recognized)
        - identity: string (name of the recognized person or "unknown")
    """
    ram_before = process.memory_info().rss
    cpu_before = process.cpu_percent()

    file = request.data
    if not file:
        return (
            jsonify(
                {
                    "error": "No image file provided",
                    "recognized": False,
                    "identity": "unknown",
                }
            ),
            400,
        )

    try:

        # Detect and crop face
        face_image = detect_and_crop_face(file)

        if face_image is None:
            ram_after = process.memory_info().rss
            cpu_after = process.cpu_percent()
            return (
                jsonify(
                    {
                        "recognized": False,
                        "identity": "unknown",
                        "message": "No face detected in the image",
                        "cpu_before": cpu_before,
                        "cpu_after": cpu_after,
                        "ram_before": ram_before,
                        "ram_after": ram_after
                    }
                ),
                200,
            )

        model = get_model()
        identity, confidence = predict_identity(model, face_image)

        recognized = identity != "unknown"
        ram_after = process.memory_info().rss
        cpu_after = process.cpu_percent()

        return jsonify({
            "recognized": recognized, 
            "identity": identity, 
            "confidence": confidence, 
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "ram_before": ram_before,
            "ram_after": ram_after
        }), 200

    except Exception as e:
        return (
            jsonify({
                "error": str(e),
                "recognized": False, 
                "identity": "unknown", 
                "confidence": 0.0,
                "cpu_before": cpu_before,
                "cpu_after": process.cpu_percent(),
                "ram_before": ram_before,
                "ram_after": process.memory_info().rss
            }), 500,
        )


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    response = {"status": "healthy"}
    return jsonify(response), 200
