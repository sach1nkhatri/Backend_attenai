from flask import Blueprint, jsonify, request
import base64
import cv2
import csv
import os
import numpy as np
from utils.image_utils import detect_faces
from utils.model_utils import load_recognizer
from utils.firebase_config import db
from datetime import datetime, timedelta
import pytz

recognize_bp = Blueprint('recognize', __name__, url_prefix="/recognize")



ATTENDANCE_CSV = "Attendance.csv"  # File path for attendance records

# ✅ Ensure CSV file exists with headers before using it
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["uid", "name", "module", "status", "timeRecorded"])  # ✅ Add CSV headers
    print("✅ Created Attendance.csv with headers.")


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


def mark_absentees():
    """Mark scheduled users as 'Absent' if they did not attend within their schedule."""
    now = datetime.now(pytz.timezone("Asia/Kathmandu"))
    current_day = now.strftime("%A")  # ✅ Get current weekday (e.g., Monday)

    print(f"🔎 Running absentee check for {current_day}...")

    schedules_ref = db.collection("schedules").stream()  # ✅ Fetch all schedules

    for schedule in schedules_ref:
        schedule_data = schedule.to_dict()
        scheduled_module = schedule_data.get("module")
        scheduled_days = schedule_data.get("workingDays", [])
        
        if current_day not in scheduled_days:
            continue  # ❌ Skip this schedule if today is not a working day

        start_time_str = schedule_data.get("startTime", "00:00")
        try:
            start_dt = datetime.strptime(start_time_str, "%H:%M")
        except ValueError:
            print(f"❌ Invalid time format in Firestore for module {scheduled_module}: {start_time_str}")
            continue

        # ✅ Convert start time to today's datetime
        start_dt = now.replace(hour=start_dt.hour, minute=start_dt.minute, second=0)
        end_dt = start_dt + timedelta(minutes=30)  # ✅ Attendance window ends after 30 minutes

        # ✅ Get all scheduled students
        scheduled_uids = [student.get("uid") for student in schedule_data.get("students", []) if isinstance(student, dict)]

        if not scheduled_uids:
            print(f"⚠️ No students scheduled for {scheduled_module}. Skipping.")
            continue  

        # ✅ Query Firestore for already marked attendance
        attendance_ref = db.collection("AttendanceRecords")
        attendance_today = attendance_ref \
            .where("module", "==", scheduled_module) \
            .where("timeRecorded", ">=", start_dt) \
            .where("timeRecorded", "<=", end_dt) \
            .stream()

        attended_uids = {record.to_dict().get("uid") for record in attendance_today}

        # ✅ Identify absentees
        absentees = [uid for uid in scheduled_uids if uid not in attended_uids]

        if absentees:
            print(f"🚨 Marking {len(absentees)} attendees as ABSENT for {scheduled_module}")

            for uid in absentees:
                try:
                    # ✅ Retrieve attendee name
                    user_name = "Unknown"
                    for student in schedule_data.get("students", []):
                        if isinstance(student, dict) and student.get("uid") == uid:
                            user_name = student.get("name", "Unknown")
                            break

                    # ✅ Save absent record to Firestore
                    new_absent_record = {
                        "uid": uid,
                        "module": scheduled_module,
                        "name": user_name,
                        "status": "Absent",
                        "timeRecorded": now  # ✅ Firestore timestamp
                    }
                    attendance_ref.add(new_absent_record)  # ✅ Save to Firestore

                    print(f"❌ {uid} marked as ABSENT for {scheduled_module}")

                except Exception as e:
                    print(f"❌ Error marking {uid} as absent: {e}")

    print("✅ Absentee marking process completed.")


@recognize_bp.route('', methods=['POST'])
def recognize_user():
    """Recognize faces, log attendance in CSV, and store in Firestore without duplicates."""
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
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d %H:%M:%S")
        today_date = now.strftime("%Y-%m-%d")  # ✅ Extract today's date

        # ✅ Step 1: Load existing attendance records from CSV (Prevents duplicate writes)
        existing_attendance = set()
        with open(ATTENDANCE_CSV, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # ✅ Skip the header row
            for row in reader:
                if len(row) >= 5:
                    existing_attendance.add((row[0], row[2], row[4][:10]))  # (uid, module, date)

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

            # ✅ Step 2: Check if user is already marked present in CSV
            if (uid, module_name, today_date) in existing_attendance:
                print(f"✅ {uid} already marked present today in module {module_name}. Skipping duplicate entry.")
                continue  # ❌ Skip writing duplicate entry

            # ✅ Step 3: Retrieve user name (🔥 FIXED)
            user_name = "Unknown"
            schedule_ref = db.collection("schedules").get()  # ✅ Get all schedules

            for schedule in schedule_ref:
                schedule_data = schedule.to_dict()
                for student in schedule_data.get("students", []):
                    if isinstance(student, dict) and student.get("uid") == uid:
                        user_name = student.get("name", "Unknown")
                        print(f"✅ Found Name for UID {uid}: {user_name}")  # Debugging Log
                        break  # Stop searching after finding the user

            # ✅ Step 4: Log attendance in CSV
            with open(ATTENDANCE_CSV, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([uid, user_name, module_name, "Present", today_str])

            print(f"✅ Attendance recorded successfully for UID {uid} in module {module_name} at {today_str}")

            # ✅ Step 5: Prevent duplicate attendance in Firestore
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)

            attendance_ref = db.collection("AttendanceRecords")
            existing_records = attendance_ref \
                .where("uid", "==", uid) \
                .where("module", "==", module_name) \
                .where("timeRecorded", ">=", start_of_day) \
                .where("timeRecorded", "<=", end_of_day) \
                .stream()

            if any(existing_records):  # ✅ If attendance already exists, SKIP saving
                print(f"✅ Attendance already exists in Firestore for UID {uid} in module {module_name}. Skipping duplicate entry.")
            else:
                try:
                    new_record = {
                        "uid": uid,
                        "module": module_name,
                        "name": user_name,
                        "status": "Present",
                        "timeRecorded": now  # ✅ Store as Firestore timestamp
                    }
                    attendance_ref.add(new_record)  # ✅ Save to Firestore
                    print(f"✅ Attendance successfully saved in Firestore for UID {uid}")

                except Exception as e:
                    print(f"❌ Firestore Error for UID {uid}: {e}")

            attendance_marked.append({"uid": uid, "module": module_name, "time": today_str})

        print("✅ Recognition process completed successfully.")

        if not attendance_marked:
            return jsonify({"message": "No attendance marked", "recognized_users": recognized_users}), 200

        return jsonify({"recognized_users": recognized_users, "attendance_marked": attendance_marked})

    except Exception as e:
        print(f"❌ ERROR in recognize_user: {str(e)}")
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500   






















































###############################THIS CODE WAS NOT USED IN THE ORIGINAL VERSION #############################

# @recognize_bp.route('', methods=['POST'])
# def recognize_user():
#     """Recognize faces, log attendance in CSV, and then store in Firestore."""
#     try:
#         print("📥 Received request for face recognition.")

#         data = request.json
#         image_data = data.get("image")

#         if not image_data:
#             print("❌ No image received in request.")
#             return jsonify({"message": "No image received"}), 400

#         recognizer = load_recognizer()
#         if recognizer is None:
#             print("❌ Face recognition model not loaded. Train the model first.")
#             return jsonify({"message": "Model not loaded. Train first."}), 500

#         frame = decode_image(image_data)
#         if frame is None:
#             print("❌ Failed to decode image from base64.")
#             return jsonify({"message": "Failed to process image"}), 400

#         print("🔍 Detecting faces...")
#         recognized_users, frame_with_boxes = detect_faces(frame, recognizer)

#         if not recognized_users:
#             print("⚠️ No recognizable faces detected in the frame.")
#             return jsonify({"message": "No recognizable faces detected"}), 200

#         attendance_marked = []
#         now = datetime.now()
#         today_str = now.strftime("%Y-%m-%d %H:%M:%S")
#         today_date = now.strftime("%Y-%m-%d")  # ✅ Extract only today's date

#         # ✅ Step 1: Load existing attendance records into memory (Prevents multiple file reads)
#         existing_attendance = set()
#         with open(ATTENDANCE_CSV, "r", newline="") as file:
#             reader = csv.reader(file)
#             next(reader, None)  # ✅ Skip the header row
#             for row in reader:
#                 if len(row) >= 5:
#                     existing_attendance.add((row[0], row[2], row[4][:10]))  # (uid, module, date)

#         for user in recognized_users:
#             uid = user["uid"]
#             confidence = user["confidence"]

#             print(f"🆔 Detected UID: {uid} with confidence: {confidence}")

#             # ✅ Skip unknown users
#             if confidence > 1000 or uid == "Unknown":
#                 print(f"❌ Skipping unknown user with UID: {uid}")
#                 continue  

#             # ✅ Check if user has a valid schedule for today
#             module_name = is_within_schedule(uid)
#             if not module_name:
#                 print(f"❌ Attendance rejected for UID {uid}. No valid schedule found.")
#                 continue  

#             # ✅ Step 2: Check if user is already marked present in CSV
#             if (uid, module_name, today_date) in existing_attendance:
#                 print(f"✅ {uid} already marked present today in module {module_name}. Skipping duplicate entry.")
#                 continue  # ❌ Skip writing duplicate entry

#             # ✅ Step 3: Retrieve user name (🔥 FIXED)
#             user_name = "Unknown"
#             schedule_ref = db.collection("schedules").get()  # ✅ Get all schedules

#             for schedule in schedule_ref:
#                 schedule_data = schedule.to_dict()
#                 for student in schedule_data.get("students", []):
#                     if isinstance(student, dict) and student.get("uid") == uid:
#                         user_name = student.get("name", "Unknown")
#                         print(f"✅ Found Name for UID {uid}: {user_name}")  # Debugging Log
#                         break  # Stop searching after finding the user

#             # ✅ Step 4: Log attendance in CSV
#             with open(ATTENDANCE_CSV, "a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([uid, user_name, module_name, "Present", today_str])

#             print(f"✅ Attendance recorded successfully for UID {uid} in module {module_name} at {today_str}")

#             # ✅ Step 5: Check if attendance already exists in Firestore
#             attendance_ref = db.collection("AttendanceRecords")
#             query = attendance_ref.where("uid", "==", uid) \
#                                   .where("module", "==", module_name) \
#                                   .where("timeRecorded", ">=", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)) \
#                                   .where("timeRecorded", "<=", datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999))

#             existing_records = query.get()

#             if not existing_records:
#                 # ✅ Step 6: Store attendance in Firestore only if no existing record is found
#                 try:
#                     new_record = {
#                         "uid": uid,
#                         "module": module_name,
#                         "name": user_name,
#                         "status": "Present",
#                         "timeRecorded": now  # ✅ Store as Firestore timestamp
#                     }

#                     attendance_ref.add(new_record)  # ✅ Save to Firestore
#                     print(f"✅ Attendance successfully saved in Firestore for UID {uid}")

#                 except Exception as e:
#                     print(f"❌ Firestore Error for UID {uid}: {e}")
#             else:
#                 print(f"✅ Attendance already exists in Firestore for UID {uid} in module {module_name}. Skipping duplicate entry.")

#             attendance_marked.append({"uid": uid, "module": module_name, "time": today_str})

#         print("✅ Recognition process completed successfully.")

#         if not attendance_marked:
#             return jsonify({"message": "No attendance marked", "recognized_users": recognized_users}), 200

#         return jsonify({"recognized_users": recognized_users, "attendance_marked": attendance_marked})

#     except Exception as e:
#         print(f"❌ ERROR in recognize_user: {str(e)}")
#         return jsonify({"message": "Internal Server Error", "error": str(e)}), 500 












# @recognize_bp.route('', methods=['POST'])
# def recognize_user():
#     """Recognize faces, log attendance in CSV, and then store in Firestore."""
#     try:
#         print("📥 Received request for face recognition.")

#         data = request.json
#         image_data = data.get("image")

#         if not image_data:
#             print("❌ No image received in request.")
#             return jsonify({"message": "No image received"}), 400

#         recognizer = load_recognizer()
#         if recognizer is None:
#             print("❌ Face recognition model not loaded. Train the model first.")
#             return jsonify({"message": "Model not loaded. Train first."}), 500

#         frame = decode_image(image_data)
#         if frame is None:
#             print("❌ Failed to decode image from base64.")
#             return jsonify({"message": "Failed to process image"}), 400

#         print("🔍 Detecting faces...")
#         recognized_users, frame_with_boxes = detect_faces(frame, recognizer)

#         if not recognized_users:
#             print("⚠️ No recognizable faces detected in the frame.")
#             return jsonify({"message": "No recognizable faces detected"}), 200

#         attendance_marked = []
#         now = datetime.now()
#         today_str = now.strftime("%Y-%m-%d %H:%M:%S")
#         today_date = now.strftime("%Y-%m-%d")  # ✅ Extract only today's date

#         # ✅ Step 1: Load existing attendance records into memory (Prevents multiple file reads)
#         existing_attendance = set()
#         with open(ATTENDANCE_CSV, "r", newline="") as file:
#             reader = csv.reader(file)
#             next(reader, None)  # ✅ Skip the header row
#             for row in reader:
#                 if len(row) >= 5:
#                     existing_attendance.add((row[0], row[2], row[4][:10]))  # (uid, module, date)

#         for user in recognized_users:
#             uid = user["uid"]
#             confidence = user["confidence"]

#             print(f"🆔 Detected UID: {uid} with confidence: {confidence}")

#             # ✅ Skip unknown users
#             if confidence > 1000 or uid == "Unknown":
#                 print(f"❌ Skipping unknown user with UID: {uid}")
#                 continue  

#             # ✅ Check if user has a valid schedule for today
#             module_name = is_within_schedule(uid)
#             if not module_name:
#                 print(f"❌ Attendance rejected for UID {uid}. No valid schedule found.")
#                 continue  

#             # ✅ Step 2: Check if user is already marked present in CSV
#             if (uid, module_name, today_date) in existing_attendance:
#                 print(f"✅ {uid} already marked present today in module {module_name}. Skipping duplicate entry.")
#                 continue  # ❌ Skip writing duplicate entry

#             # ✅ Step 3: Retrieve user name (🔥 FIXED)
#             user_name = "Unknown"
#             schedule_ref = db.collection("schedules").get()  # ✅ Get all schedules

#             for schedule in schedule_ref:
#                 schedule_data = schedule.to_dict()
#                 for student in schedule_data.get("students", []):
#                     if isinstance(student, dict) and student.get("uid") == uid:
#                         user_name = student.get("name", "Unknown")
#                         print(f"✅ Found Name for UID {uid}: {user_name}")  # Debugging Log
#                         break  # Stop searching after finding the user

#             # ✅ Step 4: Log attendance in CSV
#             with open(ATTENDANCE_CSV, "a", newline="") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([uid, user_name, module_name, "Present", today_str])

#             print(f"✅ Attendance recorded successfully for UID {uid} in module {module_name} at {today_str}")

#             # ✅ Step 5: Store attendance in Firestore
#             try:
#                 attendance_ref = db.collection("AttendanceRecords")

#                 new_record = {
#                     "uid": uid,
#                     "module": module_name,
#                     "name": user_name,
#                     "status": "Present",
#                     "timeRecorded": now  # ✅ Store as Firestore timestamp
#                 }

#                 attendance_ref.add(new_record)  # ✅ Save to Firestore
#                 print(f"✅ Attendance successfully saved in Firestore for UID {uid}")

#             except Exception as e:
#                 print(f"❌ Firestore Error for UID {uid}: {e}")

#             attendance_marked.append({"uid": uid, "module": module_name, "time": today_str})

#         print("✅ Recognition process completed successfully.")

#         if not attendance_marked:
#             return jsonify({"message": "No attendance marked", "recognized_users": recognized_users}), 200

#         return jsonify({"recognized_users": recognized_users, "attendance_marked": attendance_marked})

#     except Exception as e:
#         print(f"❌ ERROR in recognize_user: {str(e)}")
#         return jsonify({"message": "Internal Server Error", "error": str(e)}), 500


##################################COPYRIGHT TO ATTENAI #################################
