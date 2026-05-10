import numpy as np
import museval
import torch
from torch.utils.data import DataLoader
from src.datasets.musdb_segments import MusDB_segments
from src.datasets.musdb_utils import stft, istft

def evaluate_test_set(musdb_test, model, num_sources):
    """
    test_tracks: List of tuples (mixture_linear, [source1_linear, source2_linear])
    """
    results = museval.EvalStore(frames_agg='median', tracks_agg='median')

    for n in range(len(musdb_test)):
        song_segments = MusDB_segments(musdb_test[n])
        dataloader = DataLoader(song_segments, batch_size=8, shuffle=False)
        full_output_specs = {}

        waveform = np.mean(musdb_test[n].audio, dim=1, keepdims=True).T # [1, nsamples]
        waveform_length = waveform.shape[1]
        magnitude, phase = stft(waveform)

        # Get True Sources
        if num_sources == 2:
            vocal_source = np.mean(musdb_test[n].targets['vocals'].audio, dim=1, keepdims=False)
            acc_source = np.mean(musdb_test[n].targets['accompaniment'].audio, dim=1, keepdims=False)
            true_sources = np.array(vocal_source, acc_source)
        elif num_sources == 4:
            vocal_source = np.mean(musdb_test[n].targets['vocals'].audio, dim=1, keepdims=False)
            drum_source = np.mean(musdb_test[n].targets['drum'].audio, dim=1, keepdims=False)
            bass_source = np.mean(musdb_test[n].targets['bass'].audio, dim=1, keepdims=False)
            others_source = np.mean(musdb_test[n].targets['others'].audio, dim=1, keepdims=False)
            true_sources = np.array(vocal_source, drum_source, bass_source, others_source)

        for i in range(num_sources):
            full_output_specs[i] = []
        
        # 2. Model Prediction
        for i, (mixture, _) in enumerate(dataloader):
            output_specs = model(mixture)
            for i in num_sources:
                full_output_specs[i].append(output_specs[0][i].cpu())
        
        for i in num_sources:
            full_output_specs[i] = np.hstack(full_output_specs[i][:, :magnitude.shape[2]])

        # Get waveform
        source_names = ["vocals", "accompaniment"] if num_sources == 2 else ["vocals", "drums", "bass", "other"]      
        pred_sources = []
        for i, _ in enumerate(source_names):
            audio = istft(full_output_specs[i], phase, waveform_length)
            pred_sources.append(audio)
        
        # 6. Format for museval [samples, channels, sources]
        # Assuming mono audio and 1 target source for this example
        ref = np.array(true_sources).T[:, np.newaxis, :] 
        est = pred_sources.T[:, np.newaxis, :]
        
        # 5. Evaluate this specific track
        # bss_eval returns (SDR, ISR, SIR, SAR)
        scores = museval.metrics.bss_eval(ref, est)
        
        # Add to the aggregator
        results.add_track(scores)
        print(f"Track {i+1}/50 processed.")

        # Get the final median SDR across all 50 songs
        final_metrics = results.agg_frames().agg_tracks()
        return final_metrics

# results = evaluate_test_set(my_test_data, my_model)
# print(f"Global Median SDR: {results['SDR']}")