# Emotion Recognition Backend

This Flask API takes `.wav` audio input and returns emotion predictions using a pre-trained scikit-learn model.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Endpoint

POST `/analyze`

**Form-data:** `file` (audio file)

**Returns:** JSON object with emotion probabilities

---

Ensure `emotion_model.pkl` is placed in the root directory of this project.