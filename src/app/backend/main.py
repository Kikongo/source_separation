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
from typing import List
import uvicorn
import traceback

from model import SpectrogramChannelsUNet
from audio_utils import stft, istft, get_segments_spectrograms

app = FastAPI(title="Music Source Separation API")

# CORS для Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное состояние моделей
models = {}

class SeparateRequest(BaseModel):
    num_sources: int = 2  # 2 или 4

def load_model(num_sources: int):
    if num_sources not in models:
        path = f"models/unet_{num_sources}sources.pth"
        print(f"Loading model from: {os.path.abspath(path)}")
        model = SpectrogramChannelsUNet(n_sources=num_sources)
        model.load_state_dict(torch.load(f"src/app/backend/models/unet_{num_sources}sources.pth", map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        models[num_sources] = model
    return models[num_sources]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    num_sources: int = 2
):
    allowed_types = [
    "audio/wav", "audio/x-wav", 
    "audio/mpeg", "audio/mp3", 
    "audio/flac", "audio/x-flac",
    "audio/ogg"]

    # Проверка формата
    if file.content_type not in allowed_types:
        raise HTTPException(400, "Неподдерживаемый формат")
    
    # Сохранение загруженного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
        content = await file.read()
        tmp_input.write(content)
        input_path = tmp_input.name
    
    try:
        # Загрузка и предобработка аудио
        waveform, sr = torchaudio.load(input_path)
        print(waveform.shape)
        print(sr)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio_length = waveform.shape[1] / sr
        print(audio_length)
        print(int(audio_length))

        # STFT
        spectrogram, phase = stft(waveform)
        print(phase.shape)
        phase = phase.squeeze(0)

        # Get segments for model input
        segments_spectrograms = get_segments_spectrograms(waveform)
        
        # Инференс модели
        model = load_model(num_sources)
        full_output_specs = {}
        for i in range(num_sources):
            full_output_specs[i] = []
        with torch.no_grad():
            if torch.cuda.is_available():
                for segment_spectrogram in segments_spectrograms:
                    segment_spectrogram = segment_spectrogram.unsqueeze(0).cuda()
                    output_specs = model(segment_spectrogram.unsqueeze(0)) #[Batch, N_sources, freq, time]
                    for i in range(num_sources):
                        full_output_specs[i].append(output_specs[0][i])
                for i in range(num_sources):
                    full_output_specs[i] = torch.hstack(full_output_specs[i])[:, :spectrogram.shape[1]]
            else:
                for segment_spectrogram in segments_spectrograms:
                    print(segment_spectrogram.shape)

                    segment_spectrogram = segment_spectrogram.unsqueeze(0).cpu()
                    output_specs = model(segment_spectrogram.unsqueeze(0)) #[Batch, N_sources, freq, time]
                    print(f"Output {output_specs.shape}")
                    for i in range(num_sources):
                        full_output_specs[i].append(output_specs[0][i])
                for i in range(num_sources):
                    full_output_specs[i] = torch.hstack(full_output_specs[i])[:, :spectrogram.shape[2]]
        print(f" Full {full_output_specs[i].shape}")
        # Восстановление аудио для каждого источника
        result_paths = []
        source_names = ["vocals", "accompaniment"] if num_sources == 2 else ["vocals", "drums", "bass", "other"]
        
        waveform_length = audio_length * sr
        for i, name in enumerate(source_names):
            audio = istft(full_output_specs[i], phase, waveform_length)
            print(audio.shape)
            
            output_path = f"src/app/shared/{name}_{os.path.basename(input_path)}"
            sf.write(output_path, audio.numpy().T, 44100)
            result_paths.append(output_path)
        
        # Очистка входного файла
        os.unlink(input_path)
        
        return {
            "sources": source_names,
            "files": [f"/download/{os.path.basename(p)}" for p in result_paths]
        }
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/download/{filename}")
def download(filename: str):
    file_path = f"src/app/shared/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, "Файл не найден")
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)