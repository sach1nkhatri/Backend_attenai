from flask import Blueprint, Response
import cv2
from utils.image_utils import draw_faces

video_feed_bp = Blueprint('video_feed', __name__)

camera = cv2.VideoCapture(0)

@video_feed_bp.route('/live')
def live_feed():
    """Stream live video feed with face detection."""
    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = draw_faces(frame)  # Detect faces
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
