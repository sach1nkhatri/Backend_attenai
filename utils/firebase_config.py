import firebase_admin
from firebase_admin import credentials, firestore
import os

# Define the path to the Firebase credentials file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIREBASE_CRED_PATH = os.path.join(BASE_DIR, "config", "firebase_key.json")

# Ensure Firebase is initialized only once
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

# Get Firestore database instance
db = firestore.client()
