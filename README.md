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

# Deployment

Streamlit Cloud

Push your repo to GitHub

Go to https://share.streamlit.io

Connect to the repo and deploy

https://v2jopv7xkmsqkr4gk4cjwv.streamlit.app/

# Example Output

Input:

“I absolutely love this new phone 😍🔥”

Prediction:

✅ Sentiment: Positive
Confidence: 97%

<img width="1235" height="419" alt="image" src="https://github.com/user-attachments/assets/1a0c16f4-ef1c-43d2-aab8-28b5b8619d88" />


# Author

Ittyavira C Abraham

MCA (AI) — Amrita Vishwa Vidyapeetham
