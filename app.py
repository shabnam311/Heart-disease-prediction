from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Path to the models directory
MODEL_DIR = 'federated_models'

# Load a default model at startup if available
default_model = None
try:
    default_model_path = os.path.join(MODEL_DIR, 'server_1_model.pkl')
    if os.path.exists(default_model_path):
        with open(default_model_path, 'rb') as f:
            default_model = pickle.load(f)
        print("Default model (server_1) loaded successfully.")
except Exception as e:
    print(f"Error loading default model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the POST request
        data = request.json
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # We need a dataframe if the model expects feature names
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features_df = pd.DataFrame(features_array, columns=feature_names)

        # Get server choice
        server = data.get('server', 'server_1')
        model_path = os.path.join(MODEL_DIR, f'{server}_model.pkl')
        
        # Load the selected model
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model for {server} not found! Ensure training has finished.'}), 404
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Make prediction
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[0][1] # Probability of class 1 (CVD)
        
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'server_used': server
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Cardiovascular Disease Prediction Server...")
    app.run(debug=True, port=5000)
