# Contributing to Face Recognition Remote Server

Thank you for your interest in contributing to this project!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Structure

- `app/` - Flask application code
  - `__init__.py` - App initialization
  - `routes.py` - API endpoints
- `utils/` - Utility modules
  - `face_detector.py` - Face detection logic
  - `model_loader.py` - TensorFlow model handling
- `models/` - Directory for TensorFlow models
- `server.py` - Main entry point

## Testing

Run basic tests to ensure the app structure is correct:
```bash
python test_basic.py
```

For full testing with actual face recognition, you'll need to:
1. Place a TensorFlow SavedModel in `models/face_model.zip`
2. Install all dependencies from requirements.txt
3. Test with real images containing faces

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Code Style

- Follow PEP 8 guidelines
- Use docstrings for functions and modules
- Keep functions focused and single-purpose
- Handle errors gracefully

## Adding New Features

When adding new features:
- Maintain backward compatibility with the API
- Update README.md with new functionality
- Add appropriate error handling
- Consider security implications

## Security Considerations

- Never commit sensitive data or API keys
- Validate all user inputs
- Keep dependencies updated
- Use environment variables for configuration
