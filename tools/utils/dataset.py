from functools import lru_cache

import numpy as np
import pandas as pd
import random
from scipy.signal import medfilt
from torch.utils.data import Dataset


class GaitTrialInstance:
    
    def __init__(self, path_to_npz, signal_size=129):
        self.path_to_npz = path_to_npz
        
        self.signals = np.load(self.path_to_npz).reshape(-1, 51)  # L, C
        self.signal_length = self.signals.shape[0]
    
    @lru_cache(maxsize=None)
    def pad_signal(self, pad_size):
        return np.pad(self.signals, ((pad_size, pad_size), (0, 0)), mode = 'constant')

    def crop_signal_from_one_point(self, timestamp, signal_size=129): #  'left_x', 'right_x'
        # signal_size must be odd
        half_size = signal_size // 2
        pad_signal = self.pad_signal(half_size)
        crop_signal = pad_signal[timestamp: timestamp + signal_size, :]
        return crop_signal.T  # L, C -> C, L

    def crop_random_siginal_without_answer(self, signal_size=129):
        random_idx = random.randint(0, self.signal_length - 1)
        return self.crop_signal_from_one_point(random_idx, signal_size=signal_size)

    def generate_all_signal_segments_without_answer(self, signal_size=129):
        for i in range(self.signal_length):
            yield self.crop_signal_from_one_point(i, signal_size=signal_size)


class _SignalDataset(Dataset):
    def __init__(
        self,
        paths_to_npz,
    ):
        self.paths_to_npz = paths_to_npz
        self.load_instance()
    
    def load_instance(self):
        self.trial_instances = []
        for path_to_npz in self.paths_to_npz:
            instance = GaitTrialInstance(
                path_to_npz=path_to_npz,
            )
            self.trial_instances.append(instance)

    def __len__(self):
        return len(self.trial_instances)


class SignalDataset(_SignalDataset):

    def __getitem__(self, idx):
        instance = self.trial_instances[idx]
        signal = instance.crop_random_signal_with_answer()
        return signal
