#  Emotion Detector

> RISE Internship Project 7 – Tamizhan Skills  
> Built with Scikit-learn, TF-IDF, Logistic Regression, and Streamlit

A machine learning-based web app that detects the **emotion** expressed in a sentence (like joy, sadness, anger, love, etc.). This is the seventh project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build an emotion detection model that:
  - Loads and merges labeled emotion datasets from `.txt` files
  - Cleans and tokenizes the text using TF-IDF vectorization
  - Trains a **Logistic Regression classifier** for multi-class emotion prediction
  - Provides a simple and intuitive **Streamlit interface** for real-time emotion detection

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (TF-IDF + Logistic Regression)**
- **Joblib** (for model persistence)
- **Streamlit** (for interactive frontend UI)

---

##  Project Structure

```bash
emotion-detector/
├── app.py                     # Streamlit frontend for emotion prediction
├── main.py                    # Model training and evaluation script
├── requirements.txt           # All required packages
├── data/
│   ├── train.txt              # Training data (text \t label)
│   ├── val.txt                # Validation data
│   └── test.txt               # Test data
├── models/
│   └── emotion_model.pkl      # Trained Logistic Regression model
├── src/
│   └── preprocess.py          # Data loading and preprocessing logic
└── README.md                  # You're reading it 😉
```

---

## Dataset

- Source: Emotions dataset for NLP (https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- Files: train.txt, val.txt, and test.txt
- Format: Each line contains a sentence and its associated label, separated by a tab (\t)
- Emotion Labels: joy, sadness, anger, fear, love, surprise

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Multi-class classification with Logistic Regression
✅ Uses TF-IDF vectorization for efficient feature extraction
✅ High precision/recall for dominant emotions like joy and sadness
✅ Works in real-time on user input via Streamlit UI

---

## Highlights

- Handles empty and malformed lines in the dataset
- Automatically filters out stopword-only inputs
- Detects emotion from short or long text prompts
- Real-time predictions via a clean, minimal UI
- Modular design for easy upgrade to LSTM, BERT, or Hugging Face Transformers

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world use cases in mental health tech, social media monitoring, and chat-based sentiment analysis.

Built by @ShaikJasmin11
