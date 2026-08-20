# Phishing URL Detection System

A Python-based machine learning system that detects whether a URL is legitimate or potentially phishing.

## Features

- URL-based feature extraction
- Gradient Boosting Classifier
- Real-time URL prediction
- Streamlit web interface
- Model performance evaluation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradient Boosting Classifier
- Streamlit
- Joblib

## URL Features

The system extracts features such as:

- URL length
- Domain length
- Number of dots
- Number of hyphens
- Special characters
- HTTPS usage
- IP address presence
- Number of subdomains
- Suspicious keywords

## Model Performance

The model was trained using a sample of 60,000 URLs.

- Accuracy: 97.01%
- Precision: 98.30%
- Recall: 94.63%
- F1 Score: 96.43%

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt