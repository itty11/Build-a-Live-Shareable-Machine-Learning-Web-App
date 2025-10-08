import pandas as pd
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# LOAD DATA 
print("Loading Sentiment140 dataset...")
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["target", "ids", "date", "flag", "user", "text"]

# Map sentiment values
df["sentiment"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})
df = df[["text", "sentiment"]]

print(f"Loaded dataset: {df.shape[0]} samples")

# CLEAN TEXT 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"@\w+", "", text)                # Remove mentions
    text = re.sub(r"#", "", text)                   # Remove hashtags symbol
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# SPLIT DATA 
sample_df = df.sample(30000, random_state=42)  # Subsample for faster training
X_train, X_test, y_train, y_test = train_test_split(
    sample_df["clean_text"], sample_df["sentiment"], test_size=0.2, random_state=42
)

# PIPELINE 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])

# HYPERPARAMETER TUNING 
param_grid = {"clf__C": [0.1, 1.0, 10.0]}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print(f" Best Params: {grid.best_params_}")
print(f" Best CV Score: {grid.best_score_:.4f}")

# EVALUATION 
y_pred = grid.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# SAVE MODEL 
joblib.dump(grid.best_estimator_, "sentiment_model.pkl")
print("Model saved as sentiment_model.pkl")
