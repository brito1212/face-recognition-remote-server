"""Utility module for loading and using TensorFlow models."""

import numpy as np
import os
import tensorflow as tf


def load_model(model_path: str):
    """Load a TensorFlow / Keras model stored in the native `.keras` format.

    Args:
        model_path (str): Path to the `.keras` model file.

    Returns:
        tensorflow.keras.Model: Loaded model instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not `.keras`.
        RuntimeError: If loading fails.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load .keras model from {model_path}: {e}") from e

def _prepare_face(input_shape, face_image: np.ndarray) -> np.ndarray:
    """
    Resize / normalize face image to match model input shape.
    """
    # Resolve input shape (may be list for multi-input models)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    _, target_h, target_w, target_c = input_shape

    face = face_image

    # Ensure 3 channels if model expects 3
    if face.ndim == 2:  # (H, W) -> grayscale
        face = np.stack([face] * target_c, axis=-1)
    elif face.shape[-1] == 1 and target_c == 3:
        face = np.repeat(face, 3, axis=-1)

    # Resize if needed
    if face.shape[0] != target_h or face.shape[1] != target_w:
        face = tf.image.resize(face, (target_h, target_w)).numpy()

    # Normalize to 0-1 float32
    face = face.astype(np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(face, axis=0)

def predict_identity(model, face_image):
    """
    Predict the identity of a person from a face image.

    Args:
        model: Loaded TensorFlow model
        face_image: Cropped face image (numpy array)

    Returns:
        str: Predicted identity name or 'unknown'
    """

    try:

        # Add batch dimension
        face_batch = _prepare_face(model.input_shape, face_image)

        # Make prediction
        predictions = model.predict(face_batch, verbose=0)

        # Get the class with highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        # Define a confidence threshold
        confidence_threshold = 0.5

        if confidence < confidence_threshold:
            return "unknown"

        # Map class index to identity name
        # This mapping should be customized based on your model
        identity_map = {
            0: "brito",
            1: "felipe",
            2: "jorge",
            3: "vazio",
            # Add more mappings as needed
        }

        identity = identity_map.get(predicted_class, "unknown")

        return identity, float(confidence)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "unknown", 0.0
