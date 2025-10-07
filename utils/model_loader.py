"""Utility module for loading and using TensorFlow models."""

def load_model_from_zip(zip_path):
    """
    Load a TensorFlow SavedModel from a .zip file.
    
    Args:
        zip_path: Path to the .zip file containing the SavedModel
    
    Returns:
        Loaded TensorFlow model
    """
    import os
    import zipfile
    import tempfile
    import shutil
    import tensorflow as tf
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model zip file not found: {zip_path}")
    
    # Create a temporary directory to extract the model
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the SavedModel directory (usually contains 'saved_model.pb')
        model_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if 'saved_model.pb' in files:
                model_dir = root
                break
        
        if model_dir is None:
            raise ValueError("No SavedModel found in the zip file")
        
        # Load the model
        model = tf.keras.models.load_model(model_dir)
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {zip_path}: {str(e)}")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def predict_identity(model, face_image):
    """
    Predict the identity of a person from a face image.
    
    Args:
        model: Loaded TensorFlow model
        face_image: Cropped face image (numpy array)
    
    Returns:
        str: Predicted identity name or 'unknown'
    """
    import numpy as np
    
    try:
        # Normalize the image (assuming the model expects values between 0 and 1)
        face_image = face_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_image, axis=0)
        
        # Make prediction
        predictions = model.predict(face_batch, verbose=0)
        
        # Get the class with highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Define a confidence threshold
        confidence_threshold = 0.5
        
        if confidence < confidence_threshold:
            return 'unknown'
        
        # Map class index to identity name
        # This mapping should be customized based on your model
        identity_map = {
            0: 'person_0',
            1: 'person_1',
            2: 'person_2',
            # Add more mappings as needed
        }
        
        identity = identity_map.get(predicted_class, 'unknown')
        
        return identity
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 'unknown'
