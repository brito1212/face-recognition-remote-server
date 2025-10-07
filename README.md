# Face Recognition Remote Server

A Flask-based server that provides face recognition capabilities via REST API. The server receives images, detects faces, and uses a TensorFlow model to identify individuals.

## Features

- **Face Detection**: Automatically detects and crops faces from uploaded images using face_recognition library
- **Face Recognition**: Identifies individuals using a TensorFlow SavedModel
- **REST API**: Simple `/recognize` endpoint for easy integration
- **Clean Architecture**: Well-organized folder structure with separation of concerns

## Project Structure

```
face-recognition-remote-server/
├── app/
│   ├── __init__.py          # Flask app initialization
│   └── routes.py            # API endpoints
├── utils/
│   ├── __init__.py
│   ├── face_detector.py     # Face detection and cropping logic
│   └── model_loader.py      # TensorFlow model loading and prediction
├── models/
│   └── face_model.zip       # TensorFlow SavedModel (user-provided)
├── server.py                # Main entry point
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Prerequisites

- Python 3.8 or higher
- pip package manager
- A TensorFlow SavedModel packaged as a .zip file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brito1212/face-recognition-remote-server.git
cd face-recognition-remote-server
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your TensorFlow SavedModel:
   - Package your SavedModel as a .zip file
   - Place it in the `models/` directory as `face_model.zip`
   - Or set the `MODEL_ZIP_PATH` environment variable to point to your model

## Usage

### Starting the Server

Run the server with:
```bash
python server.py
```

The server will start on `http://localhost:5000` by default.

### API Endpoints

#### POST /recognize

Recognizes a face from an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with a file field named `file` containing a .jpg image

**Example using curl:**
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/recognize
```

**Example using Python requests:**
```python
import requests

url = "http://localhost:5000/recognize"
files = {'file': open('path/to/image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "recognized": true,
  "identity": "person_name"
}
```

- `recognized` (bool): Whether a face was successfully recognized
- `identity` (string): Name of the identified person, or "unknown" if not recognized

**Error Responses:**

No face detected:
```json
{
  "recognized": false,
  "identity": "unknown",
  "message": "No face detected in the image"
}
```

Invalid request:
```json
{
  "error": "Only .jpg files are supported",
  "recognized": false,
  "identity": "unknown"
}
```

#### GET /health

Health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy"
}
```

## Configuration

### Environment Variables

- `MODEL_ZIP_PATH`: Path to the TensorFlow model .zip file (default: `models/face_model.zip`)

Set environment variables before starting the server:
```bash
export MODEL_ZIP_PATH=/path/to/your/model.zip
python server.py
```

### Model Requirements

Your TensorFlow SavedModel should:
- Be packaged as a .zip file containing the SavedModel directory structure
- Include a `saved_model.pb` file
- Accept input images of shape (batch_size, 224, 224, 3) with values normalized to [0, 1]
- Output class predictions where each class represents a different person

### Customizing Identity Mapping

To map model outputs to actual names, edit the `identity_map` in `utils/model_loader.py`:

```python
identity_map = {
    0: 'John Doe',
    1: 'Jane Smith',
    2: 'Bob Johnson',
    # Add your mappings here
}
```

## Development

### Running in Development Mode

The server runs in debug mode by default when started with `python server.py`. This enables auto-reloading on code changes.

### Running in Production

For production deployment, use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```

## Dependencies

- **Flask**: Web framework
- **face-recognition**: Face detection and recognition
- **opencv-python**: Image processing
- **tensorflow**: Deep learning model inference
- **numpy**: Numerical operations

## Troubleshooting

### face-recognition Installation Issues

The `face-recognition` library requires `dlib`, which may need additional system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

**macOS:**
```bash
brew install cmake
```

**Windows:**
- Install Visual Studio Build Tools
- Or use pre-built wheels from unofficial sources

### TensorFlow Compatibility

Ensure your TensorFlow version matches the version used to create your SavedModel. Adjust the version in `requirements.txt` if needed.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
