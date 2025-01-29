import os
import cv2
import numpy as np
from PIL import Image
import re

MODEL_PATH = "TrainedModel/Trainner.yml"

def get_images_and_labels(path):
    """Extract face images and IDs from the training directory."""
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces, ids = [], []

    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('L')
            image_np = np.array(img, 'uint8')

            # ✅ Fix: Extract UID correctly using regex
            filename = os.path.basename(image_path)  # Get only filename
            match = re.search(r"_(\d+)_", filename)  # Extract numeric UID
            if match:
                user_id = int(match.group(1))  # Extract and convert to int
                faces.append(image_np)
                ids.append(user_id)
            else:
                print(f"❌ Skipping invalid filename: {filename}")

        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")

    return faces, ids

def train_recognizer():
    """Train the face recognition model with optimized parameters."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # ✅ Use more neighbors and a lower confidence threshold
    recognizer.setThreshold(50)  

    faces, ids = get_images_and_labels("TrainingImage")

    if not faces or not ids:
        print("❌ No valid training images found.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")


def load_recognizer():
    """Load the trained recognizer."""
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found! Train the model first.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer
