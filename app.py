from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load the model
model_path = r"C:\Users\cavin\Downloads\DatasetHackathon\sentiment_model"  # Update this to your model's directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define custom sentiment values
custom_sentiments = {
    "hate": -2,
    "fucking love": 4
}

def predict_sentiment(text):
    # Check for predefined phrases
    for phrase, score in custom_sentiments.items():
        if phrase in text.lower():  # Case insensitive matching
            sentiment_type = "Positive" if score > 0 else "Negative"
            return {"score": score, "type": sentiment_type}

    # If no predefined phrases are found, use the model to calculate sentiment
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    sentiment_score = logits.squeeze().item()
    sentiment_type = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    return {"score": sentiment_score, "type": sentiment_type}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_sentiment(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
