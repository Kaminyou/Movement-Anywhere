import typing as t

import numpy as np
import numpy.typing as npt


def compute_first_idx_in_positive_intervals(
    arr: npt.NDArray,
) -> t.List[int]:
    # Initialize variables
    positive_regions = []
    current_region = []
    indices_of_first_in_regions = []
    
    # Identify positive regions
    for i, num in enumerate(arr):
        if num > 0:
            current_region.append((num, i))  # Store tuple of (value, index)
        else:
            if current_region:
                positive_regions.append(current_region)
                current_region = []
    if current_region:  # Add the last region if there is one
        positive_regions.append(current_region)
    
    # Find index of first value in each positive region
    for region in positive_regions:
        values, indices = zip(*region)
        first_value_index = indices[0]
        indices_of_first_in_regions.append(first_value_index)
    
    return indices_of_first_in_regions

def filter_indices(
    original_indices: t.List[int],
    turn_mask: npt.NDArray[bool],
) -> t.List[int]:
    new_indices = []
    for index in original_indices:
        if not turn_mask[index]:
            new_indices.append(index)
    return new_indices

def find_true_index_pair(arr: npt.NDArray[bool]) -> t.Tuple[int, int]:
    # Find indices of all True values
    true_indices = np.where(arr)[0]
    
    # Initialize variables to hold the first and last True indices
    first_true = None
    last_true = None
    
    if true_indices.size > 0:  # Check if there is at least one True
        first_true = true_indices[0]
        last_true = true_indices[-1]
    
    return (first_true, last_true)

def split_indices(
    original_indices: t.List[int],
    before_criteria: int,
    after_criteria: int,
) -> t.Tuple[t.List[int], t.List[int]]:
    before = []
    after = []
    for original_index in original_indices:
        if original_index < before_criteria:
            before.append(original_index)
        if original_index > after_criteria:
            after.append(original_index)
    return before, after

def filter_indices_by_depth(
    indices: t.List[int],
    depth: npt.NDArray,
    depth_min: float,
) -> t.List[int]:
    new_indices = []
    for index in indices:
        if depth[index] < depth_min:
            continue
        new_indices.append(index)
    return new_indices

def indices_to_intervals(indices: t.List[int]) -> t.List[t.List[int]]:
    n = len(indices)
    intervals = []
    intervals_diff = []
    for i in range(n - 1):
        intervals.append([indices[i], indices[i + 1]])
        intervals_diff.append(indices[i + 1] - indices[i])
    interval_median = np.median(intervals_diff)
    intervals_new = []
    for interval, interval_diff in zip(intervals, intervals_diff):
        if interval_diff < interval_median * 0.7:
            continue
        if interval_diff > interval_median * 1.3:
            continue
        intervals_new.append(interval)
    return intervals_new

def get_gait_parameter(
    intervals: t.List[t.List[int]],
    signals: npt.NDArray,
    leg: str = 'left',
    sl_adjust: float = 1.0,
) -> t.List[t.Dict[str, float]]:
    if leg == 'left':
        idx = 2
    else:
        idx = 5
    results = []
    for interval in intervals:
        sl = abs(signals[:, idx][interval[1]] - signals[:, idx][interval[0]]) / 10 * sl_adjust # cm
        sw = np.abs(signals[:, 1] - signals[:, 4])[interval[0]] / 10  # cm
        st = (interval[1] - interval[0]) / 30  # s
        v = sl / 100 / st # m/s
        c = 1 / st * 60 # 1/s
        results.append(
            {
                'start': interval[0],
                'end': interval[1],
                'leg': leg,
                'sl': sl,
                'sw': sw,
                'st': st,
                'v': v,
                'c': c,
            },
        )
    return results

def summarize_gait_parameters(gait_parameters: t.List[t.Dict[str, float]]) -> t.Dict[str, float]:
    out = {}
    for k in ['sl', 'sw', 'st', 'v', 'c']:
        parameters = []
        for gait_parameter in gait_parameters:
            parameters.append(gait_parameter[k])
        out[k] = np.array(parameters).mean()
    return out