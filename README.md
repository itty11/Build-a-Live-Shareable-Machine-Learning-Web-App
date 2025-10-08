# Build-a-Live-Shareable-Machine-Learning-Web-App

# 💬 Twitter Sentiment Analysis using Sentiment140

This project uses the **Sentiment140 dataset (1.6M tweets)** to train a **machine learning model** that predicts tweet sentiment as **positive**, **neutral**, or **negative**.  
It also includes an **interactive Streamlit web app** for live predictions and batch CSV uploads.

##  Features

1. Clean & preprocess raw tweets
2. TF-IDF + Logistic Regression with hyperparameter tuning
3. Confusion Matrix visualization
4. Streamlit UI for:
- Live tweet prediction  
- CSV upload for bulk sentiment analysis  
- Confidence visualization bar charts

# Install Requirements

pip install -r requirements.txt

# Download Dataset

Get Sentiment140 dataset from:

https://www.kaggle.com/datasets/kazanova/sentiment140

# Train Model

python train_sentiment_model.py

# Outputs:

sentiment_model.pkl

confusion_matrix.png

# Run Web App

streamlit run app.py

# Model Info

| Component      | Description                 |
| -------------- | --------------------------- |
| **Algorithm**  | Logistic Regression         |
| **Vectorizer** | TF-IDF (10,000 features)    |
| **Evaluation** | 3-fold GridSearchCV         |
| **Classes**    | Negative, Neutral, Positive |

# Author

Ittyavira C Abraham

MCA (AI) — Amrita Vishwa Vidyapeetham
