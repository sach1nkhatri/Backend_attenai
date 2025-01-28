from flask import Blueprint, request, jsonify
from utils.file_utils import create_directories, save_user_to_csv
from utils.image_utils import crop_and_save_faces
from utils.model_utils import train_recognizer

register_bp = Blueprint('register', __name__, url_prefix="/register")
register_bp.strict_slashes = False  # Fix TypeError

@register_bp.route('', methods=['POST'])  # Fix 308 redirect issue
def register_user():
    """Register a new user, save details, and train the model."""
    data = request.json
    user_id = data.get('id')
    name = data.get('name')
    images = data.get('images')

    if not user_id or not name or not images:
        return jsonify({"message": "ID, name, and images are required."}), 400

    # Ensure directories exist
    create_directories()

    # Save user details in CSV
    save_user_to_csv(user_id, name)

    # Process and save images
    saved_count = crop_and_save_faces(user_id, name, images)

    # Train the face recognition model
    train_recognizer()

    return jsonify({"message": f"{saved_count} images processed, user saved, and model trained!"})
