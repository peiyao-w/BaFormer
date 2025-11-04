# code from https://github.com/TengdaHan/DPC/blob/master/utils/utils.py (MIT License) - many modifications/additions
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import torch
import math
from collections import defaultdict

def convert_labels(labels):
    action_borders = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
    action_borders.insert(0, -1)
    action_borders.append(len(labels) - 1)
    label_start_end = []
    for i in range(1, len(action_borders)):
        label, start, end = labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
        label_start_end.append((label, start, end))
    return label_start_end

def start_end2center_width(start_end):
    return torch.stack([start_end.mean(dim=2), start_end[:,:,1] - start_end[:,:,0]], dim=2)

def compute_offsets(time_stamps):
    #bs = time_stamps.shape[0]
    #assert bs == 1, 'Not yet implemented for larger batchsizes.'
    time_stamps.insert(0, -1)
    time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
    return time_stamps_unnormalized

def convert_labels_to_segments(labels):
    bs = labels.shape[0]
    assert bs == 1, 'Not yet implemented for larger batchsizes.'
    labels = labels[0, :]
    segments = convert_labels(labels)
    # we need to insert <sos> and <eos>
    segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
    segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
    target_labels = torch.stack([s[0] for s in segments]).unsqueeze(0) + 2
    start_end = torch.stack([torch.tensor([s[1], s[2]]) for s in segments]).unsqueeze(0).float()
    center_width = start_end2center_width(start_end)
    #start_end_norm = start_end / start_end[:,-1,-1]
    #center_width_norm = start_end2center_width(start_end_norm)
    target_durations_unnormalized = compute_offsets([s[2] for s in segments]).to(target_labels.device).unsqueeze(0)
    segments_dict = {'labels': target_labels,
                     'durations': target_durations_unnormalized,
                     'start_end': start_end.to(target_labels.device),
                     'center_width': center_width.to(target_labels.device)}
    return segments_dict