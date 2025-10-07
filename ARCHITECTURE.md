# System Architecture

## Overview

The Face Recognition Remote Server is a Flask-based REST API that processes images to detect and identify faces using TensorFlow models.

## Request Flow

```
Client Request (POST /recognize)
    |
    ├── Validate File (.jpg required)
    |
    ├── Face Detection (face_recognition + OpenCV)
    |   ├── Load image from bytes
    |   ├── Detect face locations
    |   ├── Crop face region
    |   └── Resize to 224x224
    |
    ├── Model Loading (cached)
    |   ├── Extract .zip file
    |   ├── Load TensorFlow SavedModel
    |   └── Cache for future requests
    |
    ├── Prediction
    |   ├── Normalize image (0-1 range)
    |   ├── Run model inference
    |   ├── Apply confidence threshold
    |   └── Map class to identity
    |
    └── Response
        └── JSON {recognized: bool, identity: string}
```

## Component Diagram

```
┌─────────────────────────────────────────────────┐
│              Client Application                  │
│         (curl, Python, JavaScript, etc.)        │
└────────────────┬────────────────────────────────┘
                 │ HTTP POST /recognize
                 │ (multipart/form-data)
                 ▼
┌─────────────────────────────────────────────────┐
│           Flask Server (server.py)               │
│  ┌───────────────────────────────────────┐     │
│  │      Routes (app/routes.py)           │     │
│  │  - /recognize (POST)                  │     │
│  │  - /health (GET)                      │     │
│  └───────────┬───────────────────────────┘     │
└──────────────┼─────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│Face Detector │  │  Model Loader   │
│  (utils/)    │  │    (utils/)     │
│              │  │                 │
│ - Detect     │  │ - Load .zip     │
│ - Crop       │  │ - Cache model   │
│ - Resize     │  │ - Predict       │
└──────────────┘  └─────────────────┘
       │                │
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│face_recognition│ │  TensorFlow     │
│   OpenCV     │  │   SavedModel    │
└──────────────┘  └─────────────────┘
```

## Data Flow

### Input
- **Format**: JPEG image file
- **Method**: multipart/form-data POST request
- **Field name**: `file`

### Processing Steps

1. **Image Loading**
   - Convert bytes to numpy array
   - Decode using OpenCV

2. **Face Detection**
   - Use face_recognition library (based on dlib)
   - Detect face bounding boxes
   - Extract first detected face

3. **Preprocessing**
   - Crop face region
   - Resize to 224x224 pixels
   - Convert to RGB format

4. **Model Inference**
   - Normalize pixel values (0-1)
   - Add batch dimension
   - Run model.predict()
   - Get class with highest probability

5. **Post-processing**
   - Apply confidence threshold (default: 0.5)
   - Map class index to identity name
   - Return 'unknown' if below threshold

### Output
```json
{
  "recognized": true,
  "identity": "person_name"
}
```

## Error Handling

The server handles various error conditions:

- **No file provided**: 400 Bad Request
- **Invalid file format**: 400 Bad Request
- **No face detected**: 200 OK with recognized=false
- **Model loading error**: 500 Internal Server Error
- **Prediction error**: 500 Internal Server Error

## Performance Considerations

1. **Model Caching**: The TensorFlow model is loaded once and cached in memory
2. **Lazy Imports**: Heavy libraries are imported only when needed
3. **Single Face**: Only the first detected face is processed (optimization)
4. **Batch Size**: Single image inference (can be optimized for batch processing)

## Security Features

1. **File Size Limit**: 16MB maximum upload size
2. **File Type Validation**: Only .jpg files accepted
3. **Error Sanitization**: Errors don't expose internal details to clients
4. **No Data Persistence**: Images are processed in memory, not saved

## Scalability Options

For production deployments:

1. **Use Gunicorn**: Multi-worker WSGI server
2. **Add Redis**: Cache detection results
3. **Use GPU**: Enable TensorFlow GPU support
4. **Load Balancing**: Multiple server instances behind nginx
5. **Queue System**: Async processing with Celery

## Configuration

Environment variables:
- `MODEL_ZIP_PATH`: Path to model file (default: models/face_model.zip)
- `FLASK_ENV`: Environment mode (development/production)
- `FLASK_DEBUG`: Debug mode flag

## Dependencies

Core libraries:
- **Flask**: Web framework
- **face_recognition**: Face detection (wraps dlib)
- **OpenCV**: Image processing
- **TensorFlow**: Model inference
- **NumPy**: Numerical operations
