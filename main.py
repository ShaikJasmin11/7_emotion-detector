# main.py

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.preprocess import load_txt_data

def train_emotion_model():
    df_train = load_txt_data('data/train.txt')
    df_val = load_txt_data('data/val.txt')

    X_train, y_train = df_train['text'], df_train['label']
    X_val, y_val = df_val['text'], df_val['label']

    # ✅ Build pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # ✅ Fit the model
    model.fit(X_train, y_train)

    # ✅ Predict & evaluate
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))

    # ✅ Save model
    joblib.dump(model, 'models/emotion_model.pkl')
    print("✅ Emotion model saved.")



if __name__ == '__main__':
    train_emotion_model()
