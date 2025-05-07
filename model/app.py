import warnings
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import psycopg2
from mtcnn import MTCNN
import base64
from flask_cors import CORS
import logging
from sklearn.metrics.pairwise import cosine_similarity
from waitress import serve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load TFLite model
model_path = os.path.join(os.path.dirname(__file__), "facenet_model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MTCNN for face detection
detector = MTCNN()

# Database connection configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'face_recognition'),    
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433')
}

def preprocess_image(image_bytes):
    """Convert uploaded image to model input format"""
    try:
        if isinstance(image_bytes, io.BytesIO):
            image_bytes = image_bytes.getvalue()
        
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        expected_shape = (160, 160)
        img = img.resize(expected_shape)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise ValueError("Failed to preprocess the image.") from e

@app.route('/')
def home():
    return "Face Recognition API is running! Send a POST request to /recognize with an image file"

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        img = np.array(Image.open(io.BytesIO(image_bytes)))

        detections = detector.detect_faces(img)
        if not detections:
            return jsonify({"error": "No face detected in the image."}), 400
        
        x, y, width, height = detections[0]['box']
        face = img[y:y + height, x:x + width]

        face_image = Image.fromarray(face.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        face_image.save(buffered, format="JPEG")
        buffered.seek(0)
        input_data = preprocess_image(buffered)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        new_embedding = interpreter.get_tensor(output_details[0]['index'])[0]

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if face already exists
        cursor.execute("SELECT id, name, embeddings FROM users")
        rows = cursor.fetchall()
        
        if rows:
            similarities = []
            for row in rows:
                stored_embedding = row[2]
                similarity = cosine_similarity([new_embedding], [stored_embedding])[0][0]
                similarities.append(float(similarity))
            
            threshold = 0.85
            if max(similarities) >= threshold:
                cursor.close()
                conn.close()
                return jsonify({
                    "success": False,
                    "message": "You cannot register your face twice. Thank you!"
                }), 200

        # Register new face
        name = "Sonam Tenzin"  
        cursor.execute(
            "INSERT INTO users (name, embeddings) VALUES (%s, %s)",
            (name, new_embedding.tolist())
        )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Face registered successfully"
        })
    except Exception as e:
        logger.error(f"Unhandled error during processing: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        img = np.array(Image.open(io.BytesIO(image_bytes)))

        detections = detector.detect_faces(img)
        if not detections:
            return jsonify({"error": "No face detected in the image."}), 400
        
        x, y, width, height = detections[0]['box']
        face = img[y:y + height, x:x + width]

        face_image = Image.fromarray(face.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        face_image.save(buffered, format="JPEG")
        buffered.seek(0)
        input_data = preprocess_image(buffered)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        new_embedding = interpreter.get_tensor(output_details[0]['index'])[0]

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, embeddings FROM users")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        similarities = []
        for row in rows:
            user_id, name, stored_embedding = row
            similarity = cosine_similarity([new_embedding], [stored_embedding])[0][0]
            similarities.append({
                "id": user_id,
                "name": name,
                "similarity": float(similarity)
            })

        best_match = max(similarities, key=lambda x: x['similarity'])
        threshold = 0.85
        
        if best_match["similarity"] >= threshold:
            message = f"{best_match['name']}, your attendance will be marked."
        else:
            message = "No matching face found in the database."
            
        return jsonify({
            "success": True,
            "best_match": best_match,
            "message": message,
            "all_matches": similarities
        })
    except Exception as e:
        logger.error(f"Unhandled error during processing: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
    