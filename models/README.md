# Models Directory

Place your TensorFlow SavedModel .zip file here.

## Model File

The server expects a file named `face_model.zip` by default, or you can specify a custom path using the `MODEL_ZIP_PATH` environment variable.

## Model Format

Your model should be:
- A TensorFlow SavedModel packaged as a .zip file
- The zip should contain the standard SavedModel directory structure with `saved_model.pb`

## Example Structure

When extracted, your model should look like:
```
model_directory/
├── saved_model.pb
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/ (optional)
```

Simply zip this directory and place it here as `face_model.zip`.
