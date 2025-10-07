"""Utility module for face detection and cropping using only OpenCV."""


def detect_and_crop_face(image_data):
    """Detect and crop a face from raw image bytes using OpenCV only.

    Args:
        image_data (bytes): Encoded image bytes (e.g. JPEG/PNG).

    Returns:
        numpy.ndarray | None: Cropped face image in RGB (224x224) or None if not found.
    """
    import numpy as np
    import cv2

    # Convert bytes to numpy array and decode
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")

    # Load Haar cascade (bundled with OpenCV). Using frontal default.
    cascade_path = (
        getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():  # Fallback or explicit error
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Choose the largest face (more robust when multiple)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Crop with bounds check
    h_img, w_img = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    face_bgr = image[y0:y1, x0:x1]
    if face_bgr.size == 0:
        return None

    # Resize and convert to RGB for downstream models expecting RGB
    face_bgr = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Persist cropped face for inspection/debugging
    try:  # Non-fatal if saving fails
        import os

        os.makedirs("tmp", exist_ok=True)
        save_path = os.path.join("tmp", "face_cropped.png")
        cv2.imwrite(save_path, face_bgr)  # Save in BGR (correct for imwrite)
    except Exception:
        pass

    return face_rgb
