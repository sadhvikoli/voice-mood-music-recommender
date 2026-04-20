from pathlib import Path
import os
import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_colwidth", 80)

# --------------------------------------------------
# Project paths
# --------------------------------------------------
NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "svm_model_2.0.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "svm_scaler_2.0.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "svm_label_encoder_2.0.pkl"
SPOTIFY_CSV = PROJECT_ROOT / "data" / "dataset.csv"

required_files = [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, SPOTIFY_CSV]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing required files: {missing}")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

AUDIO_FEATURES = [
    "valence",
    "energy",
    "tempo",
    "danceability",
    "acousticness",
    "instrumentalness",
    "loudness",
]

df = pd.read_csv(SPOTIFY_CSV)
df = df[["track_name", "artists", "track_genre"] + AUDIO_FEATURES].dropna()
df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)

for col in ["tempo", "loudness"]:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_max == col_min:
        df[col] = 0.0
    else:
        df[col] = (df[col] - col_min) / (col_max - col_min)

# --------------------------------------------------
# Emotion profiles
# --------------------------------------------------
EMOTION_PROFILES = {
    "happy": {
        "targets": {
            "valence": 0.85,
            "energy": 0.80,
            "tempo": 0.72,
            "danceability": 0.82,
            "acousticness": 0.20,
            "instrumentalness": 0.08,
            "loudness": 0.72,
        },
        "genre_keywords": [
            "pop", "dance", "disco", "funk", "forro", "samba",
            "salsa", "dancehall", "party", "electro", "house"
        ],
        "filters": {
            "valence_min": 0.60,
            "energy_min": 0.58,
            "danceability_min": 0.55,
        },
        "weights": {
            "similarity": 0.45,
            "valence": 0.20,
            "energy": 0.15,
            "danceability": 0.15,
            "tempo": 0.05,
        },
    },
    "sad": {
        "targets": {
            "valence": 0.20,
            "energy": 0.25,
            "tempo": 0.30,
            "danceability": 0.30,
            "acousticness": 0.70,
            "instrumentalness": 0.30,
            "loudness": 0.30,
        },
        "genre_keywords": [
            "acoustic", "ambient", "piano", "indie",
            "sad", "singer-songwriter", "rainy-day", "chill"
        ],
        "filters": {
            "valence_max": 0.42,
            "energy_max": 0.45,
        },
        "weights": {
            "similarity": 0.45,
            "acousticness": 0.20,
            "instrumentalness": 0.10,
            "valence_inverse": 0.15,
            "energy_inverse": 0.10,
            "tempo_inverse": 0.10,
        },
    },
    "angry": {
        "targets": {
            "valence": 0.30,
            "energy": 0.95,
            "tempo": 0.86,
            "danceability": 0.58,
            "acousticness": 0.10,
            "instrumentalness": 0.15,
            "loudness": 0.90,
        },
        "genre_keywords": ["rock", "metal", "punk", "hardcore", "industrial", "grunge"],
        "filters": {
            "energy_min": 0.72,
            "loudness_min": 0.55,
        },
        "weights": {
            "similarity": 0.50,
            "energy": 0.20,
            "loudness": 0.15,
            "tempo": 0.10,
            "danceability": 0.05,
        },
    },
    "fearful": {
        "targets": {
            "valence": 0.25,
            "energy": 0.55,
            "tempo": 0.50,
            "danceability": 0.35,
            "acousticness": 0.45,
            "instrumentalness": 0.35,
            "loudness": 0.45,
        },
        "genre_keywords": ["ambient", "dark", "soundtrack", "electronic", "trip-hop"],
        "filters": {"valence_max": 0.45},
        "weights": {
            "similarity": 0.50,
            "instrumentalness": 0.15,
            "acousticness": 0.10,
            "energy": 0.10,
            "tempo": 0.05,
            "loudness": 0.10,
        },
    },
    "neutral": {
        "targets": {
            "valence": 0.50,
            "energy": 0.50,
            "tempo": 0.50,
            "danceability": 0.50,
            "acousticness": 0.40,
            "instrumentalness": 0.25,
            "loudness": 0.50,
        },
        "genre_keywords": [],
        "filters": {},
        "weights": {
            "similarity": 0.60,
            "valence_balance": 0.10,
            "energy_balance": 0.10,
            "danceability": 0.10,
            "tempo": 0.10,
        },
    },
    "calm": {
        "targets": {
            "valence": 0.60,
            "energy": 0.25,
            "tempo": 0.30,
            "danceability": 0.40,
            "acousticness": 0.78,
            "instrumentalness": 0.50,
            "loudness": 0.25,
        },
        "genre_keywords": ["acoustic", "ambient", "classical", "jazz", "piano", "sleep", "chill"],
        "filters": {
            "energy_max": 0.40,
            "tempo_max": 0.45,
        },
        "weights": {
            "similarity": 0.45,
            "acousticness": 0.20,
            "instrumentalness": 0.15,
            "energy_inverse": 0.10,
            "tempo_inverse": 0.10,
        },
    },
    "disgust": {
        "targets": {
            "valence": 0.20,
            "energy": 0.55,
            "tempo": 0.50,
            "danceability": 0.40,
            "acousticness": 0.30,
            "instrumentalness": 0.20,
            "loudness": 0.60,
        },
        "genre_keywords": ["alternative", "industrial", "dark", "experimental"],
        "filters": {"energy_min": 0.40},
        "weights": {
            "similarity": 0.55,
            "energy": 0.15,
            "loudness": 0.10,
            "instrumentalness": 0.10,
            "tempo": 0.10,
        },
    },
    "surprised": {
        "targets": {
            "valence": 0.72,
            "energy": 0.78,
            "tempo": 0.68,
            "danceability": 0.68,
            "acousticness": 0.18,
            "instrumentalness": 0.10,
            "loudness": 0.72,
        },
        "genre_keywords": ["pop", "dance", "electro", "edm", "funk", "house"],
        "filters": {
            "energy_min": 0.55,
            "valence_min": 0.50,
        },
        "weights": {
            "similarity": 0.45,
            "energy": 0.18,
            "valence": 0.15,
            "danceability": 0.12,
            "tempo": 0.10,
        },
    },
}

# --------------------------------------------------
# Feature extraction and emotion prediction
# --------------------------------------------------
def extract_features(filepath: str, sr: int = 22050, duration: float = 3.0) -> np.ndarray:
    y, sr = librosa.load(filepath, sr=sr, duration=duration)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_feat = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feat = np.hstack([chroma.mean(axis=1), chroma.std(axis=1)])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_feat = np.array([centroid.mean(), centroid.std()])

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_feat = np.array([zcr.mean(), zcr.std()])

    rms = librosa.feature.rms(y=y)
    rms_feat = np.array([rms.mean(), rms.std()])

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_feat = np.array([rolloff.mean(), rolloff.std()])

    return np.hstack([
        mfcc_feat,
        chroma_feat,
        centroid_feat,
        zcr_feat,
        rms_feat,
        rolloff_feat,
    ])


def predict_emotion(audio_path: str) -> str:
    feats = extract_features(audio_path)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    pred_enc = model.predict(feats_scaled)[0]
    emotion = le.inverse_transform([pred_enc])[0]
    return emotion

# --------------------------------------------------
# Recommendation helpers
# --------------------------------------------------
def apply_emotion_filters(candidates: pd.DataFrame, emotion: str) -> pd.DataFrame:
    rules = EMOTION_PROFILES[emotion]["filters"]
    filtered = candidates.copy()

    if "valence_min" in rules:
        filtered = filtered[filtered["valence"] >= rules["valence_min"]]
    if "valence_max" in rules:
        filtered = filtered[filtered["valence"] <= rules["valence_max"]]
    if "energy_min" in rules:
        filtered = filtered[filtered["energy"] >= rules["energy_min"]]
    if "energy_max" in rules:
        filtered = filtered[filtered["energy"] <= rules["energy_max"]]
    if "danceability_min" in rules:
        filtered = filtered[filtered["danceability"] >= rules["danceability_min"]]
    if "tempo_max" in rules:
        filtered = filtered[filtered["tempo"] <= rules["tempo_max"]]
    if "loudness_min" in rules:
        filtered = filtered[filtered["loudness"] >= rules["loudness_min"]]

    if len(filtered) >= 20:
        return filtered
    return candidates


def apply_genre_filter(candidates: pd.DataFrame, emotion: str) -> pd.DataFrame:
    keywords = EMOTION_PROFILES[emotion]["genre_keywords"]
    if not keywords:
        return candidates

    genre_mask = candidates["track_genre"].astype(str).str.lower().apply(
        lambda g: any(keyword in g for keyword in keywords)
    )
    genre_filtered = candidates[genre_mask]

    if len(genre_filtered) >= 20:
        return genre_filtered
    return candidates


def compute_final_score(candidates: pd.DataFrame, emotion: str) -> pd.DataFrame:
    weights = EMOTION_PROFILES[emotion]["weights"]
    scored = candidates.copy()
    scored["final_score"] = 0.0

    for feature_name, weight in weights.items():
        if feature_name == "similarity":
            scored["final_score"] += weight * scored["similarity_score"]
        elif feature_name == "valence_inverse":
            scored["final_score"] += weight * (1 - scored["valence"])
        elif feature_name == "energy_inverse":
            scored["final_score"] += weight * (1 - scored["energy"])
        elif feature_name == "tempo_inverse":
            scored["final_score"] += weight * (1 - scored["tempo"])
        elif feature_name == "valence_balance":
            scored["final_score"] += weight * (1 - (scored["valence"] - 0.5).abs() * 2)
        elif feature_name == "energy_balance":
            scored["final_score"] += weight * (1 - (scored["energy"] - 0.5).abs() * 2)
        else:
            scored["final_score"] += weight * scored[feature_name]

    return scored.sort_values(by="final_score", ascending=False)


def diversify_recommendations(
    candidates: pd.DataFrame,
    top_n: int = 10,
    max_per_genre: int = 2,
    max_per_artist: int = 1,
) -> pd.DataFrame:
    final_rows = []
    genre_counts = {}
    artist_counts = {}

    for _, row in candidates.iterrows():
        genre = str(row["track_genre"]).strip().lower()
        artist = str(row["artists"]).strip().lower()

        if genre_counts.get(genre, 0) >= max_per_genre:
            continue
        if artist_counts.get(artist, 0) >= max_per_artist:
            continue

        final_rows.append(row)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

        if len(final_rows) == top_n:
            break

    if len(final_rows) < top_n:
        used_keys = (
            {
                (str(r["track_name"]).lower(), str(r["artists"]).lower())
                for _, r in pd.DataFrame(final_rows).iterrows()
            }
            if final_rows
            else set()
        )

        for _, row in candidates.iterrows():
            key = (str(row["track_name"]).lower(), str(row["artists"]).lower())
            if key in used_keys:
                continue
            final_rows.append(row)
            if len(final_rows) == top_n:
                break

    result = pd.DataFrame(final_rows).reset_index(drop=True)
    result.index += 1
    return result


def recommend_songs(emotion: str, top_n: int = 10) -> pd.DataFrame:
    if emotion not in EMOTION_PROFILES:
        raise ValueError(f'Emotion "{emotion}" not found.')

    targets = EMOTION_PROFILES[emotion]["targets"]
    query_vec = np.array([[targets[feature] for feature in AUDIO_FEATURES]])
    track_matrix = df[AUDIO_FEATURES].values

    similarity_scores = cosine_similarity(query_vec, track_matrix)[0]

    candidates = df.copy()
    candidates["similarity_score"] = similarity_scores

    candidates = apply_emotion_filters(candidates, emotion)
    candidates = apply_genre_filter(candidates, emotion)
    candidates = compute_final_score(candidates, emotion)
    candidates = diversify_recommendations(
        candidates,
        top_n=top_n,
        max_per_genre=2,
        max_per_artist=1,
    )

    columns = ["track_name", "artists", "track_genre", "similarity_score", "final_score"]
    candidates = candidates[columns].copy()
    candidates["similarity_score"] = candidates["similarity_score"].round(4)
    candidates["final_score"] = candidates["final_score"].round(4)

    return candidates

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def mood_music_pipeline(audio_path: str, top_n: int = 10):
    print(f"Analyzing: {os.path.basename(audio_path)}")
    print("-" * 50)

    emotion = predict_emotion(audio_path)
    print(f"Detected Emotion: {emotion.upper()}")

    print(f'\nTop {top_n} songs for "{emotion}" mood:')
    print("-" * 50)

    recs = recommend_songs(emotion, top_n=top_n)
    print(recs.to_string())

    return emotion, recs

# --------------------------------------------------
# Local test
# --------------------------------------------------
if __name__ == "__main__":
    test_file = PROJECT_ROOT / "data" / "archive" / "Actor_01" / "03-01-03-01-01-01-01.wav"
    emotion, recs = mood_music_pipeline(str(test_file), top_n=10)