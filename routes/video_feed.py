from flask import Blueprint, Response
import cv2
from utils.image_utils import draw_faces
from utils.firebase_config import db
from datetime import datetime
import pytz

video_feed_bp = Blueprint('video_feed', __name__)

camera = cv2.VideoCapture(0)

def is_schedule_available():
    """Check if there is a scheduled attendance session for the current time."""
    tz = pytz.timezone("Asia/Kathmandu")
    now = datetime.now(tz)
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M")

    schedules_ref = db.collection("schedules")
    schedules_query = schedules_ref.stream()

    for schedule in schedules_query:
        schedule_data = schedule.to_dict()

        if current_day in schedule_data.get("workingDays", []):
            start_time = schedule_data.get("startTime", "00:00")
            if start_time == current_time:
                return True  # ✅ Schedule found, start camera

    return False  # ❌ No schedule found, do not start camera


@video_feed_bp.route('/live')
def live_feed():
    """Stream live video feed with face detection, only if a schedule exists."""
    if not is_schedule_available():
        return Response("No scheduled attendance session.", status=403)

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














































# from flask import Blueprint, Response
# import cv2
# from utils.image_utils import draw_faces

# video_feed_bp = Blueprint('video_feed', __name__)

# camera = cv2.VideoCapture(0)

# @video_feed_bp.route('/live')
# def live_feed():
#     """Stream live video feed with face detection."""
#     def generate():
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             frame = draw_faces(frame)  # Detect faces
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
