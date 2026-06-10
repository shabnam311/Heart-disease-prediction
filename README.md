# Federated CVD Prediction

A decentralized, privacy-preserving machine learning system for predicting Cardiovascular Disease without exposing sensitive medical records.

### 🌐 Live Demo
You can try the AI directly in your browser:
**[Launch Application](https://shabnam311.github.io/Heart-disease-prediction/)**

---

### 🧠 Cloud Architecture
- **Frontend:** A glassmorphism HTML/JS UI hosted statically on GitHub Pages.
- **Backend:** A Python Flask API hosted on Hugging Face Spaces.
- **How it works:** When you click "Run Analysis", the GitHub UI securely pings the Hugging Face cloud brain. The AI analyzes your biometrics across 10 secure, simulated hospital nodes and returns a probability instantly.

---

### 📊 Performance Metrics
We achieved outstanding predictive accuracy using Differential Privacy, SMOTE class balancing, and Secure Aggregation:
- **Global Adaptive Accuracy**: 94.56%
- **Target F1-Score**: >94.00%

---

### 🛠️ Local Setup
If you want to run the model locally instead of using the cloud demo:
1. **Clone & Install**: `git clone https://github.com/shabnam311/Heart-disease-prediction.git` then `pip install -r requirements.txt`.
2. **Download Data**: Get the [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle and place `cardio_train.csv` inside a `data/` folder.
3. **Train Models**: Run `python train_local.py`
4. **Start API**: Run `python app.py`

### License
MIT License
