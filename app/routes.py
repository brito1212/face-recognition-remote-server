"""Flask routes for face recognition."""
from flask import Blueprint, request, jsonify
import os
from utils.face_detector import detect_and_crop_face
from utils.model_loader import load_model_from_zip, predict_identity

bp = Blueprint('main', __name__)

# Global variable to store the loaded model
_model = None
_model_path = None

def get_model():
    """Load and cache the TensorFlow model."""
    global _model, _model_path
    model_zip_path = os.environ.get('MODEL_ZIP_PATH', 'models/face_model.zip')
    
    if _model is None or _model_path != model_zip_path:
        _model = load_model_from_zip(model_zip_path)
        _model_path = model_zip_path
    
    return _model

@bp.route('/recognize', methods=['POST'])
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
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'recognized': False,
            'identity': 'unknown'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'recognized': False,
            'identity': 'unknown'
        }), 400
    
    if not file.filename.lower().endswith('.jpg'):
        return jsonify({
            'error': 'Only .jpg files are supported',
            'recognized': False,
            'identity': 'unknown'
        }), 400
    
    try:
        # Read image data
        image_data = file.read()
        
        # Detect and crop face
        face_image = detect_and_crop_face(image_data)
        
        if face_image is None:
            return jsonify({
                'recognized': False,
                'identity': 'unknown',
                'message': 'No face detected in the image'
            }), 200
        
        # Load model and predict identity
        model = get_model()
        identity = predict_identity(model, face_image)
        
        recognized = identity != 'unknown'
        
        return jsonify({
            'recognized': recognized,
            'identity': identity
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recognized': False,
            'identity': 'unknown'
        }), 500

@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200
