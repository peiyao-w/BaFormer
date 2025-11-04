import torch
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

class AverageMeter_acc:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.num_correct = 0
        self.avg = 0
        self.count = 0

    def update(self, num_correct, val, num):
        self.val = val
        self.num_correct += num_correct
        self.count += num
        self.avg = self.num_correct / self.count

class AverageMeter_f1:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_tp = torch.zeros(3)
        self.sum_fp = torch.zeros(3)
        self.sum_fn = torch.zeros(3)
        self.val = 0
        self.avg = 0


    def update(self, f_scores):
        self.val = compute_f1(f_scores)

        tp, fp, fn = f_scores
        self.sum_tp += tp
        self.sum_fp += fp
        self.sum_fn += fn
        self.avg = compute_f1((self.sum_tp, self.sum_fp, self.sum_fn))

        # precision = self.sum_tp / (self.sum_tp + self.sum_fp)
        # recall = self.sum_tp / (self.sum_tp + self.sum_fn)
        # f1 = 2.0 * (precision * recall) / (precision + recall)
        # self.avg1 = torch.nan_to_num(f1) * 100

def compute_f1(f_scores):
    tp, fp, fn = f_scores
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1) * 100
    return f1


class SumMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0

    def update(self, val):
        self.sum += val

class SumCountMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, count):
        self.sum += val
        self.count += count

    def average(self,):
        avg = self.sum/self.count
        return avg


class F1ScoreMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_tp = 0
        self.sum_fp = 0
        self.sum_fn = 0

    def update(self, tp, fp, fn):
        self.sum_tp += torch.tensor(tp)
        self.sum_fp += torch.tensor(fp)
        self.sum_fn += torch.tensor(fn)

    def average(self, ):
        precision = self.sum_tp / (self.sum_tp + self.sum_fp)
        recall = self.sum_tp / (self.sum_tp + self.sum_fn)
        f1 = 2.0 * (precision * recall) / (precision + recall)
        avg = torch.nan_to_num(f1) * 100
        return avg