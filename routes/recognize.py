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
    """Check if user is allowed to mark attendance based on scheduled weekdays."""
    
    now = datetime.now(pytz.timezone("Asia/Kathmandu"))
    current_day = now.strftime("%A")  # ✅ Get the current weekday (e.g., Monday)
    current_time = now.strftime("%H:%M")  # ✅ Get the current time (24-hour format)

    print(f"🔎 Checking attendance eligibility for UID: {uid} on {current_day} at {current_time}")

    schedules_ref = db.collection("schedules")
    schedules_query = schedules_ref.stream()  # ✅ Get all schedules

    for schedule in schedules_query:
        schedule_data = schedule.to_dict()

        print(f"📌 Found schedule in Firestore: {schedule_data}")

        # ✅ Extract UIDs from array of dictionaries
        extracted_uids = [student.get("uid") for student in schedule_data.get("students", []) if isinstance(student, dict)]
        
        print(f"🆔 Extracted UIDs: {extracted_uids}")

        if uid not in extracted_uids:
            print(f"❌ UID {uid} is NOT in the students list for {schedule_data.get('module')}")
            continue  # Skip this schedule

        # ✅ Ensure today is a valid scheduled day
        scheduled_days = schedule_data.get("workingDays", [])  # ✅ Read list directly
        if current_day not in scheduled_days:
            print(f"❌ {uid} is NOT scheduled for today ({current_day}).")
            continue

        start_time_str = schedule_data.get("startTime", "00:00")  # ✅ Get 24-hour format time
        try:
            start_dt = datetime.strptime(start_time_str, "%H:%M")  # ✅ Convert from string to datetime
        except ValueError:
            print(f"❌ Invalid time format in Firestore for {uid}: {start_time_str}")
            continue  # ✅ Skip this entry if time is invalid

        grace_period = timedelta(minutes=30)  # ✅ Allow ±30 minutes for attendance

        # ✅ Convert `start_dt` to today's date for accurate comparison
        start_dt = now.replace(hour=start_dt.hour, minute=start_dt.minute, second=0)

        print(f"🕒 Checking schedule for {uid}: Now={current_time}, Scheduled={start_time_str}, Weekday={current_day}")

        if start_dt - grace_period <= now <= start_dt + grace_period:
            print(f"✅ {uid} is within schedule for module {schedule_data['module']}")
            return schedule_data["module"]  # ✅ Return module name if valid

    print(f"❌ {uid} is NOT within schedule today.")
    return None  # ❌ User is not scheduled for attendance today





@recognize_bp.route('', methods=['POST'])
@recognize_bp.route('', methods=['POST'])
def recognize_user():
    """Recognize multiple faces and mark attendance ONLY if today is a scheduled weekday."""
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
        attendance_marked = []

        now = get_current_time()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Track already marked users to avoid duplicate attendance
        marked_users = set()

        for user in recognized_users:
            uid = user["uid"]
            confidence = user["confidence"]

            # ✅ Skip unknown users
            if confidence > 1000 or uid == "Unknown":
                print(f"❌ Skipping unknown user with UID: {uid}")
                continue  

            # ✅ Check if user has a valid schedule for today
            module_name = is_within_schedule(uid)
            if not module_name:
                print(f"❌ Attendance rejected for {uid}. No valid schedule found.")
                continue  

            print(f"📌 Marking attendance for {uid} in module {module_name} at {now_str}")

            # ✅ Retrieve name from `schedules.students`
            try:
                schedule_ref = db.collection("schedules").stream()  # Fetch all schedules
                user_name = "Unknown"

                for schedule in schedule_ref:
                    schedule_data = schedule.to_dict()
                    
                    # ✅ Iterate over students array and extract name manually
                    for student in schedule_data.get("students", []):
                        if isinstance(student, dict) and student.get("uid") == uid:
                            user_name = student.get("name", "Unknown")
                            print(f"✅ Found Name for UID {uid}: {user_name}")
                            break  # ✅ Exit loop once name is found

                print(f"🆔 Final Retrieved Name: {user_name}")

                attendance_ref = db.collection("AttendanceRecords")

                # ✅ Check if user has already been marked
                existing_attendance = attendance_ref.where("uid", "==", uid).where("module", "==", module_name).stream()
                already_marked = any(existing_attendance)

                if already_marked:
                    print(f"✅ {uid} already marked present. Skipping duplicate entry.")
                    continue  

                # ✅ Mark user as "Present"
                print(f"📝 Creating attendance record for {uid} in {module_name}")

                new_record = {
                    "uid": uid,
                    "module": module_name,
                    "name": user_name,
                    "status": "Present",
                    "timeRecorded": now_str
                }

                attendance_ref.add(new_record)
                marked_users.add(uid)
                print(f"✅ Attendance recorded in Firestore for {uid}: {new_record}")

            except Exception as e:
                print(f"❌ Error retrieving student name from Firestore for {uid}: {e}")

            attendance_marked.append({"uid": uid, "module": module_name, "time": now_str})

        return jsonify({"recognized_users": recognized_users, "attendance_marked": attendance_marked})

    except Exception as e:
        print("❌ Error in recognize_user:", str(e))
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500




