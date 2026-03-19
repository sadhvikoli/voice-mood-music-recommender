# 🎵 Mood-y: Voice-Based Mood Music Recommender

A machine learning pipeline that analyzes voice recordings to detect the speaker's emotional state and recommends music that matches their mood.

---

## Project Structure

```
voice-mood-music-recommender/
│
├── data/
│   └── README.md        # Dataset download instructions
│
├── models/              # Saved model files (generated after training)
│
├── notebooks/           # Jupyter notebooks
│
├── src/                 # Reusable Python scripts
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Datasets

> Datasets are NOT included in this repo due to size. Please download them manually before running any notebooks.

### 1. RAVDESS — Emotion Detection
- [Download from Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- After downloading, unzip and place the folder at: `data/archive/`
- Expected structure: `data/archive/Actor_XX/*.wav`

### 2. Spotify Tracks Dataset — Music Recommendation
- [Download from Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- After downloading, place the CSV at: `data/dataset.csv`

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/sadhvikoli/voice-mood-music-recommender.git
cd voice-mood-music-recommender
```

### 2. Create a virtual environment
```bash
python3 -m venv moody-env
source moody-env/bin/activate      # Mac/Linux
moody-env\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
Follow the instructions in the **Datasets** section above.

---

## Running the Project

Run the notebooks in order:

1. `01_feature_extraction.ipynb` — extracts audio features from RAVDESS, saves `X_features.npy` and `y_labels.npy`
2. `02_emotion_classification.ipynb` — trains and evaluates emotion classification models
3. `03_music_recommendation.ipynb` — builds the music recommendation system
4. `04_pipeline.ipynb` — runs the full end-to-end pipeline

> You **must** run `01_feature_extraction.ipynb` first before any other notebooks.

---

## Methodology

### Feature Extraction
Audio recordings from RAVDESS are processed using `librosa` to extract:
- **MFCCs** (40 coefficients) — mean & std
- **Chroma** (12 bins) — mean & std
- **Spectral Centroid** — mean & std
- **Zero-Crossing Rate** — mean & std
- **RMS Energy** — mean & std
- **Spectral Rolloff** — mean & std

Total: **112 features** per audio file

### Emotion Classification
Supervised learning models trained on 8 emotions: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix, cross-validation

### Music Recommendation
- Each detected emotion maps to a target Spotify audio feature profile (valence, energy, tempo, danceability, etc.)
- Cosine similarity is used to find the top-N matching tracks from 81,343 Spotify songs

---

## Requirements

See `requirements.txt`. Key libraries:
- `librosa` — audio feature extraction
- `scikit-learn` — ML models
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualization
- `joblib` — model saving

---

## Resources
- [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)