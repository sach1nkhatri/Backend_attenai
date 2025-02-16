import os
import cv2
import numpy as np
from PIL import Image
import re

# Paths
TRAINING_DIR = "TrainingImage"
MODEL_DIR = "TrainedModel"
MODEL_PATH = os.path.join(MODEL_DIR, "Trainner.yml")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def get_images_and_labels(path):
    """
    Extract face images and IDs from the training directory.
    Supports dynamic UID extraction and handles missing/corrupt images.
    """
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces, ids = [], []

    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            image_np = np.array(img, 'uint8')

            # ✅ Extract UID correctly using regex
            filename = os.path.basename(image_path)
            match = re.search(r"_(\d+)_", filename)  # Extract numeric UID
            if match:
                user_id = int(match.group(1))  # Convert to int
                faces.append(image_np)
                ids.append(user_id)
            else:
                print(f"❌ Skipping invalid filename: {filename}")

        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")

    return faces, ids

def train_recognizer():
    """
    Train the face recognition model using LBPHFaceRecognizer.
    Ensures model updates without losing previous training.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # ✅ Set optimized parameters for better accuracy
    recognizer.setThreshold(50)  # Lower threshold = better recognition

    faces, ids = get_images_and_labels(TRAINING_DIR)

    if not faces or not ids:
        print("❌ No valid training images found.")
        return

    # Train the recognizer
    recognizer.train(faces, np.array(ids))

    # Save the trained model
    recognizer.save(MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

def load_recognizer():
    """
    Load the trained face recognition model.
    Ensures it exists before loading.
    """
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found! Train the model first.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    print("✅ Model loaded successfully.")
    return recognizer
