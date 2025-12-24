import os
from sqlalchemy import create_engine

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "patient_data.db")
DB_URI = f"sqlite:///{DB_PATH}"

def get_engine():
    return create_engine(DB_URI, future=True)