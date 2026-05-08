from torch.utils.data import Dataset

class MusDB_segments(Dataset):
    def __init__(self, instance_data):

        self.mixture_segments_spectrograms = instance_data['mixture_segments_spectrograms']
        self.sources_segments_spectrograms = instance_data['vocals_segments_spectrograms']

    def __len__(self):
        return self.mixture_segments_spectrograms.shape[0]

    def __getitem__(self, ind):
        mixture_segment = self.mixture_segments_spectrograms[ind]
        sources_segment = self.sources_segments_spectrograms[ind]
        return mixture_segment, sources_segment