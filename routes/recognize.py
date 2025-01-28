from flask import Blueprint, jsonify, request
from utils.image_utils import detect_faces
from utils.model_utils import load_recognizer

recognize_bp = Blueprint('recognize', __name__, url_prefix="/recognize")
recognize_bp.strict_slashes = False  # Fix TypeError

@recognize_bp.route('', methods=['POST'])  # Fix 308 redirect issue
def recognize_user():
    """Recognize faces from webcam feed (sent by frontend)."""
    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({"message": "No image received"}), 400

    recognizer = load_recognizer()
    recognized_users = detect_faces(image_data, recognizer)

    return jsonify({"recognized_users": recognized_users})
