import cv2
import numpy as np
import base64
import os
from .file_utils import get_haarcascade_path

# Load the face detection model
detector = cv2.CascadeClassifier(get_haarcascade_path())


def crop_and_save_faces(user_id, name, images, max_faces=50):
    """Crop faces from images, enhance them, and save for training."""
    TRAINING_DIR = "TrainingImage"
    os.makedirs(TRAINING_DIR, exist_ok=True)

    saved_count = 0
    for idx, img_data in enumerate(images):
        if saved_count >= max_faces:
            break

        try:
            # ✅ Decode base64 image
            img_data = base64.b64decode(img_data.split(",")[1])  
            img_path = f"{TRAINING_DIR}/{name}_{user_id}_{idx + 1}.jpg"

            with open(img_path, "wb") as f:
                f.write(img_data)

            # ✅ Read the saved image
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Error: Could not read {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

            if len(faces) == 0:
                print(f"❌ No face detected in {img_path}. Skipping...")
                continue

            # ✅ Crop, enhance, and save faces
            for (x, y, w, h) in faces:
                cropped_face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(cropped_face, (300, 300))  # ✅ Normalize size for training
                cv2.imwrite(img_path, resized_face)  # ✅ Save cropped face
                saved_count += 1
                print(f"✅ Saved face {saved_count} to {img_path}")

                if saved_count >= max_faces:
                    break  # Stop capturing once limit is reached

        except Exception as e:
            print(f"❌ Error processing image {idx + 1}: {e}")

    return saved_count



def draw_faces(frame):
    """Detect faces and draw bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def detect_faces(frame, recognizer):
    """Detect and recognize faces in an image frame."""
    if not isinstance(frame, np.ndarray):  # Ensure valid input
        print("❌ Invalid frame format in detect_faces")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50)  # 🔧 Adjusted parameters
    )

    recognized_users = []
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # ✅ Lowering confidence threshold for better accuracy
        if conf < 65:  # 🔧 Previously 45, now allowing slightly higher confidence
            recognized_users.append({"uid": str(id), "confidence": round(conf, 2)})
        else:
            recognized_users.append({"uid": "Unknown", "confidence": round(conf, 2)})

    return recognized_users


