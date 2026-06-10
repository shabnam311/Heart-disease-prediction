from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='.')

MODEL_DIR = 'federated_models'

EXPECTED_FEATURES = [
    'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        server_id = data.get('server')
        
        if not server_id:
            return jsonify({'error': 'Server ID not provided.'}), 400

        model_path = os.path.join(MODEL_DIR, f'{server_id}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model for {server_id} not found locally.'}), 404

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Extract features
        age = float(data.get('age', 50))
        gender = int(data.get('gender', 1))
        height = float(data.get('height', 165))
        weight = float(data.get('weight', 70))
        ap_hi = float(data.get('ap_hi', 120))
        ap_lo = float(data.get('ap_lo', 80))
        cholesterol = int(data.get('cholesterol', 1))
        gluc = int(data.get('gluc', 1))
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        active = int(data.get('active', 1))
        
        # Derived
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        age_years = age

        feature_values = {
            'age_years': age_years,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'bmi': bmi,
            'pulse_pressure': pulse_pressure
        }

        df_input = pd.DataFrame([feature_values], columns=EXPECTED_FEATURES)

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            'server': server_id,
            'prediction': int(prediction),
            'probability': float(probability)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Clinical CVD Prediction Server...")
    app.run(debug=True, port=5000)
