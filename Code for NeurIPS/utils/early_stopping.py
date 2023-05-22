import numpy as np


def get_stop_point(scores, threshold=0.01):
    scores = np.concatenate(([np.min(scores[:2])], scores[2:]))
    diffs = scores[:-1] - scores[1:]
    for i, diff in zip(np.arange(1, len(scores) + 1), diffs):
        if diff < threshold:
            return i