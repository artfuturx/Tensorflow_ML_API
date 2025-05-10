import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432))
}

MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "epochs": 50,
    "verbose": 1
}

FEATURE_CONFIG = {"high_discount_treshold": 0.75, 
                  "low_amount_treshold":0.25,
                  }