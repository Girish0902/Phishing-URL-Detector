---
title: Phishing URL Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.30.0"
python_version: "3.12"
app_file: app.py
pinned: false
---




# 🛡️ AI-Based Phishing & Malicious URL Detection System

An intelligent machine learning-powered system for detecting phishing and malicious URLs with explainable risk scoring.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Problem Statement

Phishing attacks and malicious URLs are among the top attack vectors in cybersecurity, leading to credential theft, financial loss, and malware infections. Traditional rule-based detection systems fail against obfuscated and zero-day phishing attempts.

This project uses Machine Learning + Feature Engineering to automatically classify URLs as legitimate or malicious, providing risk scores and explainability.

## ✨ Features

- **Real-time URL Analysis**: Instant classification of URLs
- **Risk Scoring**: 0-100 risk score with color-coded indicators
- **Explainable AI**: Detailed breakdown of why a URL was flagged
- **Feature Engineering**: 23+ engineered features from URL structure
- **Machine Learning Models**: Random Forest classifier with high accuracy
- **Web Interface**: Clean Streamlit UI for easy interaction
- **Model Interpretability**: Feature importance visualization
- **Batch Processing**: Support for analyzing multiple URLs

## 🏗️ Architecture

```
User Input (URL)
    ↓
Feature Extraction Engine
    ↓
23+ Engineered Features
    ↓
Random Forest Classifier
    ↓
Prediction + Confidence Score
    ↓
Explainability Layer
    ↓
Risk Assessment + Recommendations
```

## 📊 Datasets

### Primary Datasets

1. **PhishTank Dataset**
   - Source: https://phishtank.com/developer_info.php
   - Format: CSV with verified phishing URLs
   - Size: 50,000+ entries

2. **Kaggle Malicious URLs Dataset**
   - Source: https://www.kaggle.com/datasets/antonyj453/urldataset
   - Features: URLs labeled as malicious/benign
   - Size: 2.4M entries

3. **UCI Phishing Websites Dataset**
   - Source: https://archive.ics.uci.edu/dataset/327/phishing+websites
   - Features: 30 attributes for phishing detection
   - Size: 11,055 instances

### Data Structure

```csv
url,label
https://www.google.com,0
https://fake-bank-login.com,1
```

Where `label` is:
- `0`: Legitimate
- `1`: Phishing/Malicious

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/phishing-url-detector.git
   cd phishing-url-detector
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (for text processing)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 📖 Usage

### Web Interface

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open browser** at `http://localhost:8501`

3. **Enter URL** and click "Analyze"

### Python API

```python
from feature_extractor import URLFeatureExtractor
import joblib

# Load model
model = joblib.load('phishing_url_detector.pkl')

# Extract features
extractor = URLFeatureExtractor()
features = extractor.extract_features("https://suspicious-site.com")

# Predict
prediction = model.predict([list(features.values())])
print("Phishing" if prediction[0] == 1 else "Legitimate")
```

## 🧠 Model Training

### Training Pipeline

1. **Data Loading & Preprocessing**
   ```python
   from train_model import load_and_preprocess_data
   df = load_and_preprocess_data('data/phishing_dataset.csv')
   ```

2. **Feature Extraction**
   ```python
   from train_model import extract_features
   X, y, feature_names = extract_features(df)
   ```

3. **Model Training**
   ```python
   from train_model import train_main_model
   model = train_main_model(X_train, y_train)
   ```

4. **Evaluation**
   ```python
   from train_model import evaluate_model
   metrics = evaluate_model(model, X_test, y_test, "Random Forest")
   ```

### Run Training

```bash
python train_model.py
```

This will:
- Load and preprocess data
- Extract features
- Train Random Forest model
- Generate evaluation metrics
- Save model as `phishing_url_detector.pkl`
- Create visualization plots

## 🌐 Deployment

### Hugging Face Spaces

1. **Create new Space** on Hugging Face
2. **Upload files**:
   - `app.py`
   - `feature_extractor.py`
   - `phishing_url_detector.pkl`
   - `requirements.txt`
   - `README.md`

3. **Set configuration**:
   - SDK: Streamlit
   - Python Version: 3.8+
   - Hardware: CPU (free tier)

4. **Deploy**: Click "Create Space"

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📚 API Reference

### URLFeatureExtractor

```python
class URLFeatureExtractor:
    def extract_features(url: str) -> dict:
        """Extract 23+ features from URL"""
    
    def extract_features_from_df(df: DataFrame, url_column: str) -> DataFrame:
        """Extract features from DataFrame"""
```

### Key Features Extracted

| Feature | Description |
|---------|-------------|
| url_length | Total characters in URL |
| num_dots | Number of dots |
| has_https | HTTPS protocol usage |
| suspicious_keywords_count | Count of phishing keywords |
| domain_length | Domain name length |
| special_chars_ratio | Ratio of special characters |

## 📈 Evaluation

### Model Performance

```
Accuracy: 0.952
Precision: 0.947
Recall: 0.931
F1-Score: 0.939
```

### Confusion Matrix

```
Predicted:     Legitimate    Phishing
Actual:
Legitimate        1850         95
Phishing            65        1820
```

### Feature Importance

Top 5 features:
1. suspicious_keywords_count (0.234)
2. url_length (0.198)
3. has_https (0.156)
4. special_chars_ratio (0.142)
5. domain_length (0.089)

## ⚠️ Limitations

- **Zero-day Detection**: Cannot detect completely new phishing patterns
- **Domain Age**: Does not check real-time domain registration age
- **False Positives**: Legitimate URLs may be flagged in rare cases
- **Language Support**: Primarily designed for English URLs
- **Real-time Updates**: Model needs periodic retraining

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PhishTank for phishing URL data
- UCI Machine Learning Repository
- Kaggle community datasets
- Scikit-learn and Streamlit communities

## 📞 Contact

For questions or collaborations:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/girish-madarkar2005)
- GitHub Issues: [Create Issue](https://github.com/Girish0902/Phishing-URL-Detector/issues)

---

**Disclaimer**: This tool is for educational and research purposes. Always verify suspicious URLs through multiple sources before taking action.
