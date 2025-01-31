# **Backend Setup Guide for AttenAI**

## **📌 Overview**
This document provides a step-by-step guide to setting up the Python Flask backend for **AttenAI**, a face recognition-based attendance system. Follow these instructions to install dependencies, configure Firebase, and run the project successfully.

---

## **1️⃣ Setting Up a Python Virtual Environment**
A virtual environment isolates dependencies, preventing conflicts with global Python packages.

### **🔹 Windows**
1. Open **Command Prompt (cmd)** or **PowerShell**.
2. Navigate to your project directory:
   ```bash
   cd path/to/your/project
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

### **🔹 macOS/Linux**
1. Open **Terminal**.
2. Navigate to your project directory:
   ```bash
   cd path/to/your/project
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

---

## **2️⃣ Activating the Virtual Environment**

### **🔹 Windows (Command Prompt)**
```bash
venv\Scripts\activate
```

### **🔹 Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

### **🔹 macOS/Linux**
```bash
source venv/bin/activate
```

✅ If activated successfully, your terminal prompt should show `(venv)` before the command line.

---

## **3️⃣ Installing Required Dependencies**
Ensure you are inside the virtual environment before running the following command:
```bash
pip install -r requirements.txt
```
This installs all required packages for Flask, Firebase, OpenCV, and face recognition.

### **🔹 Sample `requirements.txt` File**
```txt
flask
flask-cors
firebase-admin
opencv-python
numpy
pandas
Pillow
dlib
face-recognition
requests
gunicorn
```
---

## **4️⃣ Configuring Firebase Credentials**
Firebase is used to store and manage attendance records.

### **🔹 Steps to Get Firebase Credentials**
1. **Go to Firebase Console:** [https://console.firebase.google.com/](https://console.firebase.google.com/)
2. **Select Your Project** or create a new one.
3. **Go to Project Settings > Service Accounts.**
4. **Generate a new private key** (`serviceAccountKey.json`).
5. **Download the file** and place it in the `config/` folder inside your project.

### **🔹 Sample Firebase Configuration (`firebase_config.py`)**
```python
import firebase_admin
from firebase_admin import credentials, firestore
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_CRED_PATH = os.path.join(BASE_DIR, "config", "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()
```

---

## **5️⃣ Running the Flask Backend**
Once the dependencies are installed and Firebase is configured, run the backend server.

### **🔹 Option 1: Using `python app.py`**
```bash
python app.py
```

### **🔹 Option 2: Using `flask run` (Development Mode)**
```bash
flask run
```
If everything is correct, you should see an output similar to:
```bash
 * Running on http://127.0.0.1:5000/
```

---

## **6️⃣ Testing the API**
Use **Postman** or **CURL** to test the API endpoints.

### **🔹 Example: Test Face Recognition Endpoint**
```bash
curl -X POST "http://127.0.0.1:5000/recognize" -H "Content-Type: application/json" -d '{"image": "base64_encoded_image"}'
```
✅ If the backend is working correctly, it should return JSON data with recognized user details.

---

## **7️⃣ Deploying the Backend (Optional)**
For **production deployment**, use Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```
This ensures better performance compared to the built-in Flask server.

---

## **📌 Final Checklist**
✅ **Create & Activate Virtual Environment** (`venv`)
✅ **Install Dependencies** (`pip install -r requirements.txt`)
✅ **Set Up Firebase Credentials**
✅ **Run Flask Backend** (`python app.py` or `flask run`)
✅ **Test API with Postman or CURL**

🚀 **Your Backend is Now Fully Configured!** 🎯🔥

