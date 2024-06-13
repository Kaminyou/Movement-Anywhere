from functools import lru_cache
import random

import numpy as np


class GaitTrialInstance:

    def __init__(self, path_to_npz, signal_size=129):
        self.path_to_npz = path_to_npz

        self.signals = np.load(self.path_to_npz).reshape(-1, 51)  # L, C
        self.signal_length = self.signals.shape[0]
        # signal_size must be odd
        self.signal_size = signal_size

    @lru_cache(maxsize=None)
    def pad_signal(self, pad_size):
        return np.pad(self.signals, ((pad_size, pad_size), (0, 0)), mode='constant')

    def crop_signal_from_one_point(self, timestamp):
        half_size = self.signal_size // 2
        pad_signal = self.pad_signal(half_size)
        crop_signal = pad_signal[timestamp: timestamp + self.signal_size, :]
        return crop_signal.T  # L, C -> C, L

    def crop_random_siginal_without_answer(self):
        random_idx = random.randint(0, self.signal_length - 1)
        return self.crop_signal_from_one_point(random_idx)

    def generate_all_signal_segments_without_answer(self):
        for i in range(self.signal_length):
            yield self.crop_signal_from_one_point(i)
