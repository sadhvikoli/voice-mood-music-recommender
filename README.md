# 🎵 Mood-y: Voice-Based Mood Music Recommender

Mood-y is an end-to-end machine learning application that analyzes a user's voice recording, detects the underlying emotion, and recommends songs that match the detected mood.

It combines:
- **Speech emotion recognition** using audio signal features
- **Music recommendation** using Spotify track audio features
- A **React frontend + FastAPI backend** pipeline for live voice-based interaction

---

## Project Overview

The system works in 4 stages:

1. **Voice input** — A user records their voice through the web app.
2. **Emotion detection** — The audio is processed using extracted acoustic features, and a trained ML model predicts the speaker's emotion.
3. **Mood mapping** — The predicted emotion is mapped to a target music profile.
4. **Music recommendation** — Songs are recommended based on cosine similarity between the mood profile and Spotify audio features.

---

## Project Structure

```
voice-mood-music-recommender/
│
├── backend/                 # FastAPI backend
│   └── main.py
│
├── frontend/                # React frontend
│
├── data/
│   ├── archive/             # RAVDESS dataset goes here
│   ├── dataset.csv          # Spotify tracks dataset
│   └── README.md
│
├── models/                  # Saved model artifacts
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_feature_extraction.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_svm2.0.ipynb
│   ├── 06_music_recommendation.ipynb
│   ├── 07_cnn.ipynb
│   ├── RF_recommendationSystem_01.ipynb
│   ├── RF_recommendationSystem_02.ipynb
│   └── music_recommend.py
│
├── src/                     # Reusable scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

Datasets are **not included** in this repository due to their size. Please download them manually before running the project.

### 1. RAVDESS — Speech Emotion Recognition

Used for training the emotion detection model.

- **Download:** [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Extract to:** `data/archive/`

Expected structure:
```
data/archive/Actor_01/*.wav
data/archive/Actor_02/*.wav
...
```

### 2. Spotify Tracks Dataset — Music Recommendation

Used to recommend songs based on emotion-specific audio feature profiles.

- **Download:** [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Place at:** `data/dataset.csv`

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sadhvikoli/voice-mood-music-recommender.git
cd voice-mood-music-recommender
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv moody-env
source moody-env/bin/activate
```

On Windows:
```bash
moody-env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

Follow the instructions in the [Datasets](#datasets) section above.

---

## Running the Notebooks

Run notebooks **in order** — `01_feature_extraction.ipynb` must run first since later notebooks depend on the extracted features.

| Notebook | Description |
|---|---|
| `01_feature_extraction.ipynb` | Extracts features from RAVDESS audio → saves `X_features.npy`, `y_labels.npy` |
| `02_logistic_regression.ipynb` | Baseline classification model |
| `03_svm.ipynb` | SVM-based emotion classification |
| `04_random_forest.ipynb` | Random Forest classification |
| `05_svm2.0.ipynb` | Improved SVM pipeline + final model artifacts |
| `06_music_recommendation.ipynb` | Mood-based music recommendation system |

---

## Running the App

### Backend (FastAPI)

```bash
cd backend
python -m uvicorn main:app --reload
```

- API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

- App: [http://localhost:5173](http://localhost:5173)

---

## Methodology

### 1. Feature Extraction

Audio files are processed using `librosa` to extract 112 features per sample:

| Feature | Components |
|---|---|
| MFCCs | 40 coefficients (mean + std) |
| Chroma | 12 bins (mean + std) |
| Spectral centroid | mean + std |
| Zero-crossing rate | mean + std |
| RMS energy | mean + std |
| Spectral rolloff | mean + std |

### 2. Emotion Classification

The model is trained on 8 emotions from RAVDESS: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

Models explored: Logistic Regression, SVM, Random Forest. The final pipeline uses **SVM 2.0** as the best-performing model.

Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix, cross-validation.

### 3. Music Recommendation

The predicted emotion is mapped to a target Spotify audio profile using features like `valence`, `energy`, `tempo`, `danceability`, `acousticness`, `instrumentalness`, and `loudness`. Recommendations are generated via cosine similarity across 81,000+ tracks.

---

## Example Pipeline

```
Voice Recording (.wav)
        ↓
Audio Feature Extraction (librosa)
        ↓
Emotion Prediction (SVM 2.0)
        ↓
Mood Profile Mapping
        ↓
Cosine Similarity → Spotify Tracks
        ↓
Top-N Songs Returned
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| Frontend | React |
| Audio Processing | librosa, soundfile |
| ML | scikit-learn, joblib |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## Future Improvements

- Improve emotion detection with deep learning (CNN / LSTM)
- Add Spotify playback integration or song previews
- Deploy online
- Add user history and personalized recommendations
- Improve frontend UI and animations

---

## Resources

- [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)

---

## Authors

**Sadhvi Koli** — [github.com/sadhvikoli](https://github.com/sadhvikoli)
**Aayushi Kadam** — [github.com/aayu3hi](https://github.com/aayu3hi)
**Aparnaa Senthilnathan** — [github.com/AparCode](https://github.com/AparCode)
