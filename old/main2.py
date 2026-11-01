from dotenv import load_dotenv
import os
import firebase_admin
from firebase_admin import firestore, credentials

load_dotenv()

firekey = os.getenv("FIREBASE_CONFIG")

print("firekey:", firekey)

if not firebase_admin._apps:
    cred = credentials.Certificate(firekey)
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("Firebase connected successfully")