"""Utility module for face detection and cropping using only OpenCV."""


def detect_and_crop_face(
    image_data,
    target_size=(160, 160),
    min_face_rel=0.10,   # minimum face size relative to min(image width, height)
    margin_rel=0.25,     # padding around detected face relative to face size
    debug=True,
):
    """Detect and crop a face from raw image bytes using OpenCV only.

    Args:
        image_data (bytes): Encoded image bytes (e.g. JPEG/PNG).
        target_size (tuple[int, int]): Output size (w, h), defaults to (160, 160).
        min_face_rel (float): Minimum face size as a fraction of min(image dims).
        margin_rel (float): Extra margin to include around the face.
        debug (bool): If True, saves debug crops into tmp/.

    Returns:
        numpy.ndarray | None: Cropped face image in RGB (target_size) or None if not found.
    """
    import numpy as np
    import cv2
    import math

    # Decode
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")

    h_img, w_img = image.shape[:2]
    if h_img < 10 or w_img < 10:
        return None

    # Work on a smaller copy for speed, keep scale to map back boxes
    max_detect_side = 960  # tune for latency vs accuracy
    scale = 1.0
    if max(h_img, w_img) > max_detect_side:
        scale = max_detect_side / float(max(h_img, w_img))
        det_img = cv2.resize(image, (int(w_img * scale), int(h_img * scale)), interpolation=cv2.INTER_AREA)
    else:
        det_img = image.copy()

    # Grayscale + CLAHE to stabilize detection in varied lighting
    gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Load cascades (haar + lbp). Try HAAR first, then LBP as fallback.
    def load_cascade(filename):
        base = getattr(cv2.data, "haarcascades", None)
        if base:
            path = base + filename
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
        # Some installations have lbpcascades in a sibling folder
        base_lbp = getattr(cv2.data, "lbpcascades", None)
        if base_lbp:
            path = base_lbp + filename
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
        return None

    haar = load_cascade("haarcascade_frontalface_default.xml")
    lbp = load_cascade("lbpcascade_frontalface.xml")

    detectors = [c for c in [haar, lbp] if c is not None]
    if not detectors:
        raise RuntimeError("Could not load OpenCV face cascades (haar/lbp)")

    # Dynamic minSize based on image size
    min_side = min(det_img.shape[:2])
    min_size = max(24, int(min_side * min_face_rel))

    faces_all = []
    for cascade in detectors:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # finer scale to improve recall
            minNeighbors=5,
            minSize=(min_size, min_size),
        )
        if len(faces) > 0:
            faces_all.extend(faces)

    if len(faces_all) == 0:
        return None

    # Choose face by size and centrality
    cx_img = det_img.shape[1] / 2.0
    cy_img = det_img.shape[0] / 2.0

    def score_face(f):
        x, y, w, h = f
        area = w * h
        cx = x + w / 2.0
        cy = y + h / 2.0
        # normalized distance from image center
        dn = math.hypot((cx - cx_img) / cx_img, (cy - cy_img) / cy_img)
        return area - 0.3 * dn * area  # penalize off-center slightly

    x, y, w, h = max(faces_all, key=score_face)

    # Map back to original image coordinates if scaled
    if scale != 1.0:
        inv = 1.0 / scale
        x = int(round(x * inv))
        y = int(round(y * inv))
        w = int(round(w * inv))
        h = int(round(h * inv))

    # Add margin and clamp
    pad = int(round(max(w, h) * margin_rel))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)

    # Make crop square by expanding the smaller side within bounds
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w != crop_h:
        if crop_w < crop_h:
            diff = crop_h - crop_w
            x0 = max(0, x0 - diff // 2)
            x1 = min(w_img, x1 + math.ceil(diff / 2))
        else:
            diff = crop_w - crop_h
            y0 = max(0, y0 - diff // 2)
            y1 = min(h_img, y1 + math.ceil(diff / 2))

    face_bgr = image[y0:y1, x0:x1]
    if face_bgr.size == 0:
        return None

    # Resize and convert to RGB
    face_bgr = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    if debug:
        try:
            import os
            os.makedirs("tmp", exist_ok=True)
            cv2.imwrite(os.path.join("tmp", "face_cropped_dbg.png"), face_bgr)
        except Exception:
            pass

    return face_rgb
