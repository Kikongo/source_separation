import torch
import torchaudio
import numpy as np
import soundfile as sf
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery.result import AsyncResult
import uvicorn
import traceback

from model import SpectrogramChannelsUNet
from audio_utils import stft, istft, get_segments_spectrograms
from tasks import separate_task, celery_app

app = FastAPI(title="Music Source Separation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = [
    "audio/wav", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/flac", "audio/x-flac",
    "audio/ogg",
]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    num_sources: int = 2
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported audio format")

    if num_sources not in (2, 4):
        raise HTTPException(400, "num_sources must be 2 or 4")

    # Save upload to shared volume so Celery worker can read it
    os.makedirs("/app/shared/uploads", exist_ok=True)
    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir="/app/shared/uploads"
    ) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    task = separate_task.delay(input_path, num_sources)
    return {"job_id": task.id}


@app.get("/status/{job_id}")
def status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    if result.state == "PENDING":
        return {"state": "pending"}
    elif result.state == "PROGRESS":
        return {"state": "processing", "info": result.info}
    elif result.state == "SUCCESS":
        return {"state": "success", "result": result.result}
    elif result.state == "FAILURE":
        return {"state": "failed", "error": str(result.info)}
    return {"state": result.state}


@app.get("/download/{filename}")
def download(filename: str):
    # Prevent path traversal
    filename = os.path.basename(filename)
    file_path = f"/app/shared/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)