from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
CORS(app) # Allow GitHub Pages to communicate with the local backend

MODEL_DIR = 'federated_models'

EXPECTED_FEATURES = [
    'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure'
]

# Cache models globally at startup
MODELS = {}
GLOBAL_METADATA = None

def load_models():
    global GLOBAL_METADATA
    if not os.path.exists(MODEL_DIR):
        return
        
    meta_path = os.path.join(MODEL_DIR, 'secure_aggregation_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            GLOBAL_METADATA = json.load(f)
            
    for file in os.listdir(MODEL_DIR):
        if file.endswith('_model.pkl'):
            server_id = file.replace('_model.pkl', '')
            with open(os.path.join(MODEL_DIR, file), 'rb') as f:
                MODELS[server_id] = pickle.load(f)
                
load_models()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        server_id = data.get('server')
        
        if not server_id:
            return jsonify({'error': 'Server ID not provided.'}), 400

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

        if server_id == 'global':
            # TECHNIQUE 4: SECURE AGGREGATION
            if not GLOBAL_METADATA:
                return jsonify({'error': 'Global secure aggregation metadata not found.'}), 404
                
            valid_servers = GLOBAL_METADATA['valid_servers']
            adaptive_weights = GLOBAL_METADATA['adaptive_weights']
            
            global_prob = 0.0
            for sid in valid_servers:
                if sid not in MODELS:
                    continue
                model = MODELS[sid]
                prob = model.predict_proba(df_input)[0][1]
                weight = adaptive_weights[sid]
                global_prob += (prob * weight)
                
            probability = float(global_prob)
            prediction = 1 if probability > 0.5 else 0
            
        else:
            if server_id not in MODELS:
                return jsonify({'error': f'Model for {server_id} not found locally.'}), 404

            model = MODELS[server_id]
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
    app.run(host='0.0.0.0', port=5000)
