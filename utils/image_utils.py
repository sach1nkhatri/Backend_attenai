import cv2
import numpy as np
import base64
import os
from .file_utils import get_haarcascade_path

# Load the face detection model
detector = cv2.CascadeClassifier(get_haarcascade_path())

def crop_and_save_faces(user_id, name, images):
    """Crop faces from images and save them for training."""
    TRAINING_DIR = "TrainingImage"
    os.makedirs(TRAINING_DIR, exist_ok=True)

    saved_count = 0
    for idx, img_data in enumerate(images):
        img_data = base64.b64decode(img_data.split(",")[1])  # Decode base64
        img_path = f"{TRAINING_DIR}/{name}.{user_id}.{idx + 1}.jpg"

        with open(img_path, "wb") as f:
            f.write(img_data)

        # Crop face
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cropped_face = gray[y:y+h, x:x+w]
            cv2.imwrite(img_path, cropped_face)  # Overwrite with cropped face
            saved_count += 1
            break  # Only save the first detected face per image

    return saved_count

def draw_faces(frame):
    """Detect faces and draw bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame 

def detect_faces(image_data, recognizer):
    """Detect and recognize faces in a frame."""
    image_data = base64.b64decode(image_data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    recognized_users = []
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 50:  # Accept only if confidence is below 50
            recognized_users.append({"id": id, "confidence": round(conf, 2)})
        else:
            recognized_users.append({"id": "Unknown", "confidence": round(conf, 2)})

    return recognized_users
