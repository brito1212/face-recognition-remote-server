"""Utility module for face detection and cropping."""

def detect_and_crop_face(image_data):
    """
    Detect and crop the face from an image.
    
    Args:
        image_data: Binary image data (bytes)
    
    Returns:
        numpy.ndarray: Cropped face image in RGB format, or None if no face detected
    """
    import numpy as np
    import cv2
    import face_recognition
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face locations using face_recognition
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 0:
        return None
    
    # Get the first detected face
    top, right, bottom, left = face_locations[0]
    
    # Crop the face
    face_image = rgb_image[top:bottom, left:right]
    
    # Resize to a standard size (e.g., 224x224 for most models)
    face_image = cv2.resize(face_image, (224, 224))
    
    return face_image
