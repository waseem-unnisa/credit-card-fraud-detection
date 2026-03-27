from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../models/fraud_model.pkl')

@app.route('/')
def home():
    return "Credit Card Fraud Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    
    return jsonify({'Fraud Prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)