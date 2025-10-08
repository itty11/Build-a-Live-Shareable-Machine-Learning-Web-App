import streamlit as st
import joblib
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

# PAGE CONFIG 
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="wide")

st.title("💬 Twitter Sentiment Analyzer")
st.write("Analyze tweet sentiment (positive, neutral, or negative) using a trained ML model.")

# LOAD MODEL 
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

# CLEAN FUNCTION 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# USER INPUT 
user_input = st.text_area("✍️ Enter a tweet to analyze:", height=120)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        clean = clean_text(user_input)
        prediction = model.predict([clean])[0]
        st.subheader(f"Predicted Sentiment: {prediction}")
        
        # Show confidence scores
        proba = model.predict_proba([clean])[0]
        df_prob = pd.DataFrame({
            "Sentiment": model.classes_,
            "Confidence": proba * 100
        }).sort_values(by="Confidence", ascending=False)
        
        st.bar_chart(df_prob.set_index("Sentiment"))
    else:
        st.warning("Please enter a tweet text before analyzing.")

# BATCH UPLOAD 
st.divider()
st.header("📂 Batch Sentiment Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        df["clean_text"] = df["text"].apply(clean_text)
        df["predicted_sentiment"] = model.predict(df["clean_text"])
        st.dataframe(df.head(10))
        st.download_button("⬇️ Download Predictions", df.to_csv(index=False), "predictions.csv")
