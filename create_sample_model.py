"""
Sample script showing how to create a compatible TensorFlow model and package it as .zip

This is just a template. You should replace this with your actual model training code.
"""
import tensorflow as tf
import numpy as np
import zipfile
import os
import shutil

def create_sample_model():
    """Create a simple sample model for demonstration purposes."""
    # Define a simple CNN model for face recognition
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for 3 people
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model_as_zip(model, output_path='models/face_model.zip'):
    """Save a TensorFlow model as a .zip file."""
    # Create temporary directory for SavedModel
    temp_dir = '/tmp/saved_model_temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Save the model
    print(f"Saving model to {temp_dir}...")
    model.save(temp_dir)
    
    # Create zip file
    print(f"Creating zip file at {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"Model saved successfully to {output_path}")

def main():
    """Main function to create and package a sample model."""
    print("Creating sample face recognition model...")
    print("NOTE: This is a template model without training.")
    print("Replace this with your actual trained model.\n")
    
    # Create the model
    model = create_sample_model()
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Save as zip
    save_model_as_zip(model)
    
    print("\n✓ Sample model created successfully!")
    print("\nIMPORTANT: This is an untrained model for demonstration only.")
    print("For actual face recognition, you need to:")
    print("1. Collect and label face images")
    print("2. Train the model on your dataset")
    print("3. Package the trained model using this script")
    print("\nTo use this sample model for testing the server:")
    print("  python server.py")

if __name__ == '__main__':
    # Check if TensorFlow is installed
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        main()
    except ImportError:
        print("ERROR: TensorFlow is not installed.")
        print("Install it with: pip install tensorflow")
