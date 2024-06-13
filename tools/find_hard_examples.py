import argparse
import glob
import os

import numpy as np
import torch

from utils.dataset import GaitTrialInstance, SignalDataset
from utils.model import SignalNet
from utils.process import evaluate


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', default=-1, help='How many data to find for active learning')
    args = parser.parse_args()
    return args


def main():
    args = arguments()

    model = SignalNet(num_of_class=2)
    model.load_state_dict(torch.load('backend/algorithms/gait_basic/gait_study_semi_turn_time/weights/semi_vanilla_v2/gait-turn-time.pth', map_location=torch.device('cpu')))

    paths_to_npz = []
    for folder in os.listdir('/data/'):
        potential_paths_to_npz = glob.glob(os.path.join('/data/', folder, 'out', '3d', '*.mp4.npy'))
        if len(potential_paths_to_npz) != 1:
            raise ValueError('multiple npy found')
        path_to_npz = potential_paths_to_npz[0]
        paths_to_npz.append(path_to_npz)

    eval_dataset = []
    for path_to_npz in paths_to_npz:
        instance = GaitTrialInstance(path_to_npz=path_to_npz)
        eval_dataset.append(instance)

    _, probss = evaluate(epoch=0, eval_dataset=eval_dataset, model=model, device='cpu', return_prob=True)

    diff_means = []
    for probs in probss:
        diff = abs(probs[:,0] - probs[:,1])
        for criteria in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            sub_probs = diff < criteria
            if sum(sub_probs) == 0:
                continue
            diff_mean = diff[sub_probs].mean() #
            break
        diff_means.append(diff_mean)
    sorted_idx = np.argsort(diff_means)

    count = 0
    for idx in sorted_idx:
        print(f'Number {idx + 1}; certainty={diff_means[idx]}; path={paths_to_npz[idx]}')
        count += 1
        if args.number != -1 and count >= args.number:
            break

if __name__ == '__main__':
    main()
