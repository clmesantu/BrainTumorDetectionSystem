from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import base64
import os
import logging
import secrets
from chaos_encryption import generate_key, encrypt_image, decrypt_image
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


app = Flask(__name__)

# Logging setup
log_messages = []
class ListHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))

handler = ListHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

# Paths
model_dir = r'C:\Users\Santhosh S\Downloads\tumor_detection with encryption\model'
model_path = os.path.join(model_dir, 'cnn_svm_model.pkl')

# Load the trained pipeline (includes both scaler and SVM)
pipeline = joblib.load(model_path)

# In-memory encrypted images
stored_images = {}

def extract_features(image):
    hog = cv2.HOGDescriptor()
    features = hog.compute(image).flatten()
    logging.debug(f'Extracted features: {features}')
    return features

def detect_tumor(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))  # Match training size
    features = resized_image.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    prediction = pipeline.predict(features)

    if prediction[0] == 1:
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)
        mask = np.zeros_like(image)
        mask[:, :, 1] = thresh
        highlighted_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted_image, contours, -1, (0, 255, 0), 2)

        tumor_area = sum(cv2.contourArea(c) for c in contours)
        total_area = gray_image.shape[0] * gray_image.shape[1]
        tumor_percentage = (tumor_area / total_area) * 100

        return prediction[0], highlighted_image, tumor_percentage
    else:
        return prediction[0], image, 0.0

def stage(prediction):
    return 'Malignant' if prediction == 1 else 'Benign'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    log_messages.clear()
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    try:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'})

        key = generate_key()
        encrypted = encrypt_image(image, key)
        image_id = secrets.token_hex(8)
        stored_images[image_id] = {'encrypted': encrypted, 'key': key}

        return jsonify({
            'message': 'Image encrypted. Use the key to decrypt and analyze.',
            'image_id': image_id,
            'encryption_key': key
        })
    except Exception as e:
        logging.error(f'Error processing image: {e}')
        return jsonify({'error': str(e), 'logs': log_messages})

@app.route('/decrypt-analyze', methods=['POST'])
def decrypt_analyze():
    data = request.json
    image_id = data.get('image_id')
    user_key = data.get('key')

    if not image_id or not user_key:
        return jsonify({'error': 'Missing image ID or key'})

    stored = stored_images.get(image_id)
    if not stored:
        return jsonify({'error': 'Invalid image ID'})

    if stored['key'] != user_key:
        return jsonify({'error': 'Invalid decryption key'})

    try:
        decrypted_img = decrypt_image(stored['encrypted'], user_key)
        result, processed_img, tumor_percentage = detect_tumor(decrypted_img)
        result_text = 'Tumor Detected' if result == 1 else 'No Tumor'
        tumor_stage = stage(result)
        cancer_chance = f"{min(max(tumor_percentage, 10), 95):.2f}%" if result == 1 else "0%"

        _, original_img_encoded = cv2.imencode('.png', decrypted_img)
        original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

        _, processed_img_encoded = cv2.imencode('.png', processed_img)
        processed_img_base64 = base64.b64encode(processed_img_encoded).decode('utf-8')

        return jsonify({
            'diagnosis': result_text,
            'tumor_stage': tumor_stage,
            'tumor_percentage': f"{tumor_percentage:.2f}",
            'cancer_chance': cancer_chance,
            'original_image': original_img_base64,
            'image': processed_img_base64,
            'logs': log_messages
        })

    except Exception as e:
        logging.error(f'Decryption/Analysis failed: {e}')
        return jsonify({'error': str(e), 'logs': log_messages})

if __name__ == '__main__':
    app.run(debug=True)
