import pickle
import typing as t

import numpy as np
import numpy.typing as npt
import torch

from .dataset import InferenceTrialData, inference_one_trial
from .model import SignalNet
from .utils import (
    compute_first_idx_in_positive_intervals, filter_indices,
    filter_indices_by_depth, find_true_index_pair, get_gait_parameter,
    indices_to_intervals, split_indices, summarize_gait_parameters,
)


def depth_simple_inference(
    detectron_2d_single_person_keypoints_path: str,
    rendered_3d_single_person_keypoints_path: str,
    height: float,
    depth_pretrained_path: str,
    turn_time_mask_path: str,
    device: str = 'cpu',
) -> t.Tuple[t.Dict[str, float], t.List[t.Dict[str, t.Any]]]:
    model = SignalNet(in_channels=86, num_of_class=6)
    model.load_state_dict(torch.load(depth_pretrained_path))
    model.to(device)
    model.eval()

    trial_data = InferenceTrialData(
        detectron_2d_single_person_keypoints_path=detectron_2d_single_person_keypoints_path,
        rendered_3d_single_person_keypoints_path=rendered_3d_single_person_keypoints_path,
        height=height,
        use_height=True,
        signal_type='2d+3d',
    )

    out = inference_one_trial(trial_data, model, device)
    out_re = out * np.array([1000, 500, 10000, 1000, 500, 10000])

    # f'/home/kaminyou/dev/PathoOpenGait-dev/backend/real_data/{trial_id}/out/{trial_id}-tt.pickle'
    with open(turn_time_mask_path, 'rb') as f:
        turn_time_mask = np.array(pickle.load(f), dtype=bool)

    positive_max_indices = compute_first_idx_in_positive_intervals(
        out_re[:, 2] - out_re[:, 5],
    ) # right
    negative_max_indices = compute_first_idx_in_positive_intervals(
        out_re[:, 5] - out_re[:, 2],
    ) # left

    filtered_positive_max_indices = filter_indices(positive_max_indices, turn_time_mask) # right
    filtered_negative_max_indices = filter_indices(negative_max_indices, turn_time_mask) # left

    turn_interval_pair = find_true_index_pair(turn_time_mask)

    if turn_interval_pair == (None, None):
        print('cannot determine turn interval')
        turn_interval_pair = (len(turn_time_mask) // 2, len(turn_time_mask) // 2)

    right_forward_indices, left_backward_indices = split_indices(
        filtered_positive_max_indices, turn_interval_pair[0], turn_interval_pair[1],
    )  # right
    left_forward_indices, right_backward_indices = split_indices(
        filtered_negative_max_indices, turn_interval_pair[0], turn_interval_pair[1],
    )  # left

    right_forward_indices = filter_indices_by_depth(
        right_forward_indices,
        out_re[:, 5],
        depth_min=1500,
    )
    left_backward_indices = filter_indices_by_depth(
        left_backward_indices,
        out_re[:, 2],
        depth_min=1500,
    )
    left_forward_indices = filter_indices_by_depth(
        left_forward_indices,
        out_re[:, 2],
        depth_min=1500,
    )
    right_backward_indices = filter_indices_by_depth(
        right_backward_indices,
        out_re[:, 5],
        depth_min=1500,
    )

    right_intervals = indices_to_intervals(right_forward_indices) + indices_to_intervals(right_backward_indices)
    left_intervals = indices_to_intervals(left_forward_indices) + indices_to_intervals(left_backward_indices)

    gait_parameters = get_gait_parameter(right_intervals, out_re, leg='right') + get_gait_parameter(left_intervals, out_re, leg='left')
    final_output = summarize_gait_parameters(gait_parameters)

    return final_output, gait_parameters
