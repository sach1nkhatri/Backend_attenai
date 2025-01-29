import cv2
import numpy as np
import base64
from flask import Blueprint, jsonify, request
from utils.image_utils import detect_faces
from utils.model_utils import load_recognizer

recognize_bp = Blueprint('recognize', __name__, url_prefix="/recognize")

def decode_image(image_data):
    """Convert base64-encoded image to OpenCV format."""
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])  # Decode base64
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Convert to OpenCV image
        return img
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None

@recognize_bp.route('', methods=['POST'])
def recognize_user():
    """Recognize faces from a base64-encoded image sent by the frontend."""
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"message": "No image received"}), 400

        # ✅ Decode the base64 image before passing it to OpenCV
        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"message": "Failed to process image"}), 400

        recognizer = load_recognizer()
        if recognizer is None:
            return jsonify({"message": "Model not found. Train first."}), 500

        recognized_users = detect_faces(frame, recognizer)

        return jsonify({"recognized_users": recognized_users})

    except Exception as e:
        print("❌ Error in recognize_user:", str(e))
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500
