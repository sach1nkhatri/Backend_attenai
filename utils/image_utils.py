import cv2
import numpy as np
import base64
import os
from .file_utils import get_haarcascade_path

# Load the face detection model
detector = cv2.CascadeClassifier(get_haarcascade_path())

# Directory for storing training images
TRAINING_DIR = "TrainingImage"

def crop_and_save_faces(user_id, name, images, max_faces=100, retrain=False):
    """
    Crop faces, apply histogram equalization, and save for training.
    If retraining, appends new images instead of replacing old ones.
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)
    user_folder = os.path.join(TRAINING_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    # ✅ Check existing images count
    existing_images = len(os.listdir(user_folder)) if os.path.exists(user_folder) else 0
    saved_count = existing_images if retrain else 0  # If retraining, start from existing count

    for idx, img_data in enumerate(images):
        if saved_count >= max_faces:
            break

        try:
            img_data = base64.b64decode(img_data.split(",")[1])
            img_path = os.path.join(user_folder, f"{user_id}_{saved_count + 1}.jpg")

            with open(img_path, "wb") as f:
                f.write(img_data)

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ✅ Apply histogram equalization
            equalized = cv2.equalizeHist(gray)

            faces = detector.detectMultiScale(equalized, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 0:
                continue

            for (x, y, w, h) in faces:
                face = equalized[y:y+h, x:x+w]

                # ✅ Save improved images
                resized_face = cv2.resize(face, (300, 300))
                cv2.imwrite(img_path, resized_face)

                saved_count += 1

                if saved_count >= max_faces:
                    break

        except Exception as e:
            print(f"❌ Error processing image {idx + 1}: {e}")

    return saved_count


def detect_faces(frame, recognizer):
    """
    Detect and recognize faces with dynamic confidence adjustment.
    """
    if not isinstance(frame, np.ndarray):
        print("❌ Invalid frame format in detect_faces")
        return [], frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40), maxSize=(400, 400)
    )

    recognized_users = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (300, 300))

        try:
            id, conf = recognizer.predict(face)

            # ✅ Adjust confidence threshold dynamically
            distance_factor = 1 - (w / frame.shape[1])  # Approximate distance factor
            threshold = 60 + (distance_factor * 20)  # Higher confidence needed for distant faces

            if conf > threshold:
                print(f"❌ Confidence too high ({conf}), skipping.")
                continue

            recognized_users.append({"uid": str(id), "confidence": round(conf, 2)})

            # ✅ Draw bounding box with color based on confidence
            color = (0, 255, 0) if conf < threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"ID: {id} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            print(f"❌ Error recognizing face: {e}")

    return recognized_users, frame


def draw_faces(frame):
    """
    Detect faces and draw bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box

    return frame
