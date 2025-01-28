# import os
# import pandas as pd
# from datetime import datetime

# def create_directories():
#     """Ensure required directories exist."""
#     os.makedirs("TrainingImage", exist_ok=True)
#     os.makedirs("TrainingImageLabel", exist_ok=True)
#     os.makedirs("StudentDetails", exist_ok=True)

# def save_user_to_csv(user_id, name):
#     """Save user details to a CSV file."""
#     csv_path = "StudentDetails/StudentDetails.csv"
#     if not os.path.exists(csv_path):
#         df = pd.DataFrame(columns=["ID", "Name", "RegistrationTime"])
#         df.to_csv(csv_path, index=False)

#     df = pd.read_csv(csv_path)
#     df = pd.concat([df, pd.DataFrame([[user_id, name, datetime.now()]], columns=df.columns)], ignore_index=True)
#     df.to_csv(csv_path, index=False)

# def get_haarcascade_path():
#     """Return the path to the Haarcascade XML file."""
#     return "haarcascade_frontalface_default.xml"
import os
import pandas as pd
from datetime import datetime
def create_directories():
    """Ensure required directories exist."""
    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("TrainedModel", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)

def get_haarcascade_path():
    """Return the Haarcascade XML file path."""
    return "haarcascade_frontalface_default.xml"
def save_user_to_csv(user_id, name):
    """Save user details to a CSV file."""
    csv_path = "StudentDetails/StudentDetails.csv"

    # Create CSV if it doesn't exist
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["ID", "Name", "RegistrationTime"])
        df.to_csv(csv_path, index=False)

    # Append new user data
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([[user_id, name, datetime.now()]], columns=df.columns)], ignore_index=True)
    df.to_csv(csv_path, index=False)