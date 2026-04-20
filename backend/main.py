from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMP_DIR = BASE_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
sys.path.append(str(NOTEBOOKS_DIR))

from music_recommend import mood_music_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running"}

@app.post("/recommend")
async def recommend(audio: UploadFile = File(...)):
    file_path = TEMP_DIR / audio.filename

    with open(file_path, "wb") as f:
        f.write(await audio.read())

    emotion, recs = mood_music_pipeline(str(file_path), top_n=10)

    return {
        "emotion": emotion,
        "recommendations": recs.to_dict(orient="records")
    }