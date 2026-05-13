import os
import torch
import torchaudio
import soundfile as sf
from celery import Celery

from model import SpectrogramChannelsUNet
from audio_utils import stft, istft, get_segments_spectrograms

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,          # results kept 1 hour
    worker_prefetch_multiplier=1, # one task at a time per worker (ML is heavy)
    task_acks_late=True,          # re-queue if worker crashes mid-task
)

# Module-level model cache — loaded once per worker process
_models: dict = {}

def load_model(num_sources: int) -> SpectrogramChannelsUNet:
    if num_sources not in _models:
        path = f"/app/models/unet_{num_sources}sources.pth"
        model = SpectrogramChannelsUNet(n_sources=num_sources)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _models[num_sources] = model
    return _models[num_sources]


@celery_app.task(bind=True, name="tasks.separate_task")
def separate_task(self, input_path: str, num_sources: int):
    try:
        self.update_state(state="PROGRESS", meta={"step": "loading audio"})
        print(waveform.shape)
        waveform, sr = torchaudio.load(input_path)
        if sr != 44100:
            waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        orig_length = waveform.shape[1]

        self.update_state(state="PROGRESS", meta={"step": "computing spectrogram"})
        spectrogram, phase = stft(waveform)
        # phase: [freq, time]  (squeezed below for use with istft)
        phase = phase.squeeze(0)

        segments_spectrograms = get_segments_spectrograms(waveform)

        self.update_state(state="PROGRESS", meta={"step": "running model"})
        model = load_model(num_sources)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        full_output_specs: dict[int, list] = {i: [] for i in range(num_sources)}

        # NEW — all segments in one batched forward pass
        BATCH_SIZE = 2   # tune this to your GPU VRAM

        all_outputs = []   # will collect [batch, n_sources, freq, time] tensors
        orig_spec_len = spectrogram.shape[2]

        for batch_start in range(0, len(segments_spectrograms), BATCH_SIZE):
            batch = segments_spectrograms[batch_start : batch_start + BATCH_SIZE]
            # batch: [B, freq, time]  →  add channel dim  →  [B, 1, freq, time]
            batch = batch.unsqueeze(1).to(device)
            with torch.no_grad():
                out = model(batch)    # [B, n_sources, freq, time]
            all_outputs.append(out.cpu())

        all_outputs = torch.cat(all_outputs, dim=0)  # [n_segments, n_sources, freq, time]

        for i in range(num_sources):
            # stack along time axis, then trim to original spectrogram length
            full_output_specs[i] = torch.hstack(
                [all_outputs[s][i] for s in range(all_outputs.shape[0])]
            )[:, :orig_spec_len]


        self.update_state(state="PROGRESS", meta={"step": "reconstructing audio"})
        source_names = (
            ["vocals", "accompaniment"] if num_sources == 2
            else ["vocals", "drums", "bass", "other"]
        )

        os.makedirs("/app/shared", exist_ok=True)
        base = os.path.splitext(os.path.basename(input_path))[0]
        result_files = []

        for i, name in enumerate(source_names):
            audio = istft(full_output_specs[i], phase, orig_length)
            out_path = f"/app/shared/{name}_{base}.wav"
            sf.write(out_path, audio.numpy().T, 44100)
            result_files.append(f"/download/{os.path.basename(out_path)}")

        # Clean up the upload
        try:
            os.unlink(input_path)
        except OSError:
            pass

        return {"sources": source_names, "files": result_files}

    except Exception as exc:
        # Clean up on failure too
        try:
            os.unlink(input_path)
        except OSError:
            pass
        raise self.retry(exc=exc, max_retries=0)  # don't retry ML failures silently