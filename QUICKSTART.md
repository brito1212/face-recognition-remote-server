# Quick Start Guide

## Installation in 3 Steps

1. **Clone and setup**
   ```bash
   git clone https://github.com/brito1212/face-recognition-remote-server.git
   cd face-recognition-remote-server
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model**
   - Place your TensorFlow SavedModel as `models/face_model.zip`

## Run the Server

```bash
python server.py
```

Server will be available at: http://localhost:5000

## Test the API

### Health Check
```bash
curl http://localhost:5000/health
```

### Recognize a Face
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/recognize
```

## Expected Response

```json
{
  "recognized": true,
  "identity": "person_name"
}
```

Or if no face detected:
```json
{
  "recognized": false,
  "identity": "unknown",
  "message": "No face detected in the image"
}
```

## Common Issues

### face_recognition won't install?
- **Ubuntu/Debian**: `sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev`
- **macOS**: `brew install cmake`
- **Windows**: Install Visual Studio Build Tools

### Model not found?
Set the path explicitly:
```bash
export MODEL_ZIP_PATH=/path/to/your/model.zip
python server.py
```

### Port already in use?
Edit `server.py` and change the port:
```python
app.run(host='0.0.0.0', port=8080, debug=True)
```

## Using Docker (Easiest)

```bash
# Place your model in models/face_model.zip first
docker-compose up -d
```

Server will be at: http://localhost:5000

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Check [example_usage.py](example_usage.py) for Python client examples
