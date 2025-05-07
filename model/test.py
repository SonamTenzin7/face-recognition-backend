import psycopg2

DB_CONFIG = {
    'dbname': 'face_recognition',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': 5433
}

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("Database connection successful!")
    conn.close()
except Exception as e:
    print(f"Database connection failed: {str(e)}")