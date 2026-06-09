# Federated CVD Prediction

A decentralized, privacy-preserving machine learning system for Cardiovascular Disease (CVD) prediction across multiple hospital servers.

## Features
- **Federated Learning Architecture**: Trains models across 10 distinct simulated hospital data silos without centralizing sensitive medical records.
- **Ultra-Aggressive Feature Engineering**: Automatically extracts up to 80 non-linear polynomial features and interaction terms per server.
- **Robust to Data Issues**: Specifically designed to handle local data poisoning, label noise, measurement outliers, and class imbalance using SMOTE and aggressive filtering.
- **Premium User Interface**: A dynamic, interactive Web UI designed with glassmorphism to let users test predictions against different federated server models.
- **Flask Inference Backend**: Serves the trained `.pkl` models securely for real-time predictions.

## Repository Structure
```
federated-cvd-prediction/
│
├── train_local.py                ← Local training script (converted from Jupyter Notebook)
├── models/                       ← Trained federated model files (.pkl)
├── templates/
│   └── index.html                ← Premium Web UI for predictions
├── app.py                        ← Flask inference backend
├── requirements.txt              ← Python dependencies
└── README.md                     ← Project documentation
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/federated-cvd-prediction.git
   cd federated-cvd-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Federated Models (Locally)**:
   Ensure you have the CSV datasets inside your local data folder (or update paths in `train_local.py`).
   ```bash
   python train_local.py
   ```
   *Note: This process trains robust ensembles over 15 rounds and will take time depending on your CPU.*

4. **Run the Web App UI**:
   Once models are trained and saved in the `models/` (or `federated_models/`) directory:
   ```bash
   python app.py
   ```
   Navigate to `http://127.0.0.1:5000` in your browser.

## Performance Metrics
*(To be updated after the current 15-round training completes)*
- **Target Accuracy**: 92%+
- **Target F1-Score**: 95%+

## License
MIT License
