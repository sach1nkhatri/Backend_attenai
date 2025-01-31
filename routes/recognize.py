from flask import Blueprint, jsonify, request
import base64
import cv2
import numpy as np
from utils.image_utils import detect_faces
from utils.model_utils import load_recognizer
from utils.firebase_config import db
from datetime import datetime, timedelta
import pytz

recognize_bp = Blueprint('recognize', __name__, url_prefix="/recognize")

# ✅ Store recently marked users to prevent duplicate detections
recent_attendance = {}

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

def get_current_time():
    """Get the current time in a readable format (UTC+5:45 for Nepal)."""
    tz = pytz.timezone("Asia/Kathmandu")
    return datetime.now(tz)

def is_within_schedule(uid):
    """Check if the user is allowed to mark attendance based on scheduled weekdays & time."""

    now = datetime.now(pytz.timezone("Asia/Kathmandu"))
    current_day = now.strftime("%A")  # ✅ Get current weekday (e.g., Monday)
    current_time = now.strftime("%H:%M")  # ✅ Get current time (24-hour format)

    print(f"🔎 Checking schedule for UID: {uid} on {current_day} at {current_time}")

    schedules_ref = db.collection("schedules")
    schedules_query = schedules_ref.stream()  # ✅ Get all schedules

    valid_schedules = []  # ✅ List to store valid schedules

    for schedule in schedules_query:
        schedule_data = schedule.to_dict()

        # ✅ Extract UIDs from `students` array
        extracted_uids = [student.get("uid") for student in schedule_data.get("students", []) if isinstance(student, dict)]

        if uid not in extracted_uids:
            continue  # ❌ Skip if user is not in this schedule

        # ✅ Check if today is a scheduled working day
        scheduled_days = schedule_data.get("workingDays", [])
        if current_day not in scheduled_days:
            continue  # ❌ Skip if today is not in the schedule

        start_time_str = schedule_data.get("startTime", "00:00")  # ✅ Default time format HH:MM
        try:
            start_dt = datetime.strptime(start_time_str, "%H:%M")  # ✅ Convert from string to datetime
        except ValueError:
            print(f"❌ Invalid time format in Firestore for {uid}: {start_time_str}")
            continue  # ❌ Skip if time format is incorrect

        grace_period = timedelta(minutes=30)  # ✅ Allow ±30 minutes for attendance

        # ✅ Convert `start_dt` to today's date for accurate comparison
        start_dt = now.replace(hour=start_dt.hour, minute=start_dt.minute, second=0)

        # ✅ Check if user is within attendance window
        if start_dt - grace_period <= now <= start_dt + grace_period:
            valid_schedules.append((schedule_data["module"], start_dt))

    if valid_schedules:
        # ✅ Sort by time and return the earliest valid schedule
        valid_schedules.sort(key=lambda x: x[1])
        selected_module = valid_schedules[0][0]
        print(f"✅ {uid} is within schedule for module: {selected_module}")
        return selected_module

    print(f"❌ {uid} is NOT within schedule today.")
    return None  # ❌ No valid schedule found





@recognize_bp.route('', methods=['POST'])
def recognize_user():
    """Recognize multiple faces and mark attendance ONLY if today is a scheduled weekday."""
    try:
        print("📥 Received request for face recognition.")
        
        data = request.json
        image_data = data.get("image")

        if not image_data:
            print("❌ No image received in request.")
            return jsonify({"message": "No image received"}), 400

        recognizer = load_recognizer()
        if recognizer is None:
            print("❌ Face recognition model not loaded. Train the model first.")
            return jsonify({"message": "Model not loaded. Train first."}), 500

        frame = decode_image(image_data)
        if frame is None:
            print("❌ Failed to decode image from base64.")
            return jsonify({"message": "Failed to process image"}), 400

        print("🔍 Detecting faces...")
        recognized_users, frame_with_boxes = detect_faces(frame, recognizer)

        if not recognized_users:
            print("⚠️ No recognizable faces detected in the frame.")
            return jsonify({"message": "No recognizable faces detected"}), 200

        attendance_marked = []
        now = get_current_time()
        today_str = now.strftime("%Y-%m-%d")  # ✅ Extract today's date
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        marked_users = set()

        for user in recognized_users:
            uid = user["uid"]
            confidence = user["confidence"]

            print(f"🆔 Detected UID: {uid} with confidence: {confidence}")

            # ✅ Skip unknown users
            if confidence > 1000 or uid == "Unknown":
                print(f"❌ Skipping unknown user with UID: {uid}")
                continue  

            # ✅ Check if user has a valid schedule for today
            module_name = is_within_schedule(uid)
            if not module_name:
                print(f"❌ Attendance rejected for UID {uid}. No valid schedule found.")
                continue  

            print(f"✅ Attendance approved for UID {uid} in module {module_name} at {now_str}")

            # ✅ Retrieve name from `schedules.students`
            try:
                schedule_ref = db.collection("schedules").stream()
                user_name = "Unknown"

                for schedule in schedule_ref:
                    schedule_data = schedule.to_dict()

                    for student in schedule_data.get("students", []):
                        if isinstance(student, dict) and student.get("uid") == uid:
                            user_name = student.get("name", "Unknown")
                            break  

                attendance_ref = db.collection("AttendanceRecords")

                # ✅ Check if user has already been marked **for today**
                existing_attendance = attendance_ref.where("uid", "==", uid).where("module", "==", module_name).where("timeRecorded", ">=", today_str).limit(1).stream()
                
                already_marked = any(existing_attendance)

                if already_marked:
                    print(f"✅ {uid} already marked present today. Skipping duplicate entry.")
                    continue  

                # ✅ Mark user as "Present"
                print(f"📝 Creating attendance record for UID {uid} in module {module_name}")

                new_record = {
                    "uid": uid,
                    "module": module_name,
                    "name": user_name,
                    "status": "Present",
                    "timeRecorded": now_str
                }

                attendance_ref.add(new_record)
                marked_users.add(uid)
                print(f"✅ Attendance recorded successfully for UID {uid}: {new_record}")

            except Exception as e:
                print(f"❌ Error retrieving student name from Firestore for UID {uid}: {e}")

            attendance_marked.append({"uid": uid, "module": module_name, "time": now_str})

        print("✅ Recognition process completed successfully.")
        return jsonify({"recognized_users": recognized_users, "attendance_marked": attendance_marked})

    except Exception as e:
        print(f"❌ ERROR in recognize_user: {str(e)}")
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500




