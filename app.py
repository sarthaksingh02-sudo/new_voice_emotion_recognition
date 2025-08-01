from flask_cors import CORS
from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib  # use joblib instead of pickle
import os
from werkzeug.utils import secure_filename

# Load the model
model = joblib.load("emotion_model.pkl")

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio = request.files['file']
    filename = secure_filename(audio.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(filepath)

    try:
        features = extract_features(filepath)
        proba = model.predict_proba(features)[0]
        labels = model.classes_
        result = {label: round(score * 100, 2) for label, score in zip(labels, proba)}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True)
