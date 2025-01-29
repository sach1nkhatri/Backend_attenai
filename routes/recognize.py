from flask import Blueprint, jsonify, request
import base64
import cv2
import numpy as np
from utils.image_utils import detect_faces
from utils.model_utils import load_recognizer

recognize_bp = Blueprint('recognize', __name__, url_prefix="/recognize")

def decode_image(image_data):
    """Convert base64-encoded image to OpenCV format."""
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        return None

@recognize_bp.route('', methods=['POST'])
def recognize_user():
    """Recognize multiple faces from a base64-encoded image."""
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"message": "No image received"}), 400

        recognizer = load_recognizer()
        if recognizer is None:
            return jsonify({"message": "Model not loaded. Train first."}), 500

        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"message": "Failed to process image"}), 400

        recognized_users, frame_with_boxes = detect_faces(frame, recognizer)

        # ✅ Encode the processed frame back to send to the frontend
        _, buffer = cv2.imencode(".jpg", frame_with_boxes)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"recognized_users": recognized_users, "image": frame_base64})

    except Exception as e:
        print("❌ Error in recognize_user:", str(e))
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500
