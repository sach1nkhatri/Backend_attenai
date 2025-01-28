import cv2
import numpy as np
import os
from PIL import Image

MODEL_PATH = "TrainedModel/Trainner.yml"

def train_recognizer():
    """Train the face recognition model."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels("TrainingImage")
    if not faces or not ids:
        return
    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_PATH)

def load_recognizer():
    """Load the trained recognizer."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer

def get_images_and_labels(path):
    """Extract face images and IDs from the training directory."""
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, ids = [], []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        image_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split('.')[1])
        faces.append(image_np)
        ids.append(id)

    return faces, ids
