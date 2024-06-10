import typing as t
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


class InferenceTrialData:

    signal_types = {'2d', '3d', '2d+3d'}

    def __init__(
        self,
        detectron_2d_single_person_keypoints_path: str,
        rendered_3d_single_person_keypoints_path: str,
        height: t.Optional[int] = None,
        use_height: bool = False,
        signal_size: int = 129,
        signal_type: str = '2d+3d',
    ):
        if signal_type not in self.signal_types:
            raise ValueError
        self.signal_type = signal_type
        self.signal_size = signal_size
        self.height = height
        self.use_height = use_height

        self.detectron_2d_single_person_keypoints_path = detectron_2d_single_person_keypoints_path
        self.rendered_3d_single_person_keypoints_path = rendered_3d_single_person_keypoints_path

        self._parse_detectron_2d_single_person_keypoints()
        self._parse_rendered_3d_single_person_keypoints()

        self._collect_input_signal()

    def _parse_detectron_2d_single_person_keypoints(self):
        data = np.load(self.detectron_2d_single_person_keypoints_path, allow_pickle=True)
        k = list(data.f.positions_2d.item().keys())[0]
        self.detectron_2d_keypoints = data.f.positions_2d.item()[k]['custom'][0]  # T, 17, 2
        self.detectron_2d_keypoints_normalized = self.detectron_2d_keypoints / np.array([1080, 1920])  # noqa

    def _parse_rendered_3d_single_person_keypoints(self):
        data = np.load(self.rendered_3d_single_person_keypoints_path)
        self.rendered_3d_keypoints = data  # T, 17, 3

    def _collect_input_signal(self):
        signal_2d = self.detectron_2d_keypoints_normalized.reshape(-1, 34)  # T, 34
        signal_3d = self.rendered_3d_keypoints.reshape(-1, 51)  # T, 51
        if self.signal_type == '2d':
            self.signals = signal_2d
        elif self.signal_type == '3d':
            self.signals = signal_3d
        elif self.signal_type == '2d+3d':
            self.signals = np.hstack((signal_2d, signal_3d))  # T, 85
        else:
            raise ValueError
        if self.use_height:
            if self.height is None:
                raise ValueError
            heights = np.full((self.signals.shape[0], 1), self.height) / 200
            self.signals = np.concatenate((self.signals, heights), axis=1)

        self.signal_length = self.signals.shape[0]

    @lru_cache(maxsize=None)
    def pad_signal(self, pad_size):
        return np.pad(self.signals, ((pad_size, pad_size), (0, 0)), mode='constant')

    def crop_signal_from_one_point(self, timestamp: int):
        # signal_size must be odd
        half_size = self.signal_size // 2
        pad_signal = self.pad_signal(half_size)
        crop_signal = pad_signal[timestamp: timestamp + self.signal_size, :]
        return crop_signal.T  # L, C -> C, L

    def generate_all_signal_segments_without_answer(self):
        for i in range(self.signal_length):
            yield self.crop_signal_from_one_point(i)


def inference_one_trial(
    trial_data: InferenceTrialData,
    model: nn.Module,
    device: str,
):
    signals = []
    for signal in trial_data.generate_all_signal_segments_without_answer():
        signals.append(signal)
    signals = np.stack(signals)
    signals = torch.FloatTensor(signals)
    signals = signals.to(device)
    model.eval()
    with torch.no_grad():
        out = model(signals)
    out = out.cpu().numpy()
    return out
