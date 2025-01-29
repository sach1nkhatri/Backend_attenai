from flask import Blueprint, request, jsonify
import os
from utils.file_utils import create_directories, save_user_to_csv
from utils.image_utils import crop_and_save_faces
from utils.model_utils import train_recognizer

register_bp = Blueprint('register', __name__, url_prefix="/register")

@register_bp.route('', methods=['POST'])
def register_user():
    """Register a new user, save details, and train the model."""
    try:
        data = request.json
        user_id = data.get('id')
        name = data.get('name')
        images = data.get('images')

        if not user_id or not name or not images:
            return jsonify({"message": "ID, name, and images are required."}), 400

        create_directories()

        save_user_to_csv(user_id, name)

        saved_count = crop_and_save_faces(user_id, name, images)

        if saved_count < 10:
            return jsonify({"message": "Face detection failed. Ensure proper lighting and face visibility."}), 400

        train_recognizer()

        return jsonify({"message": f"{saved_count} images processed, user saved, and model trained!"})

    except Exception as e:
        print("Error in register_user:", str(e))
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500
