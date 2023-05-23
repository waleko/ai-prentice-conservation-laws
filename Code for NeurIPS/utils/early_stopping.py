import numpy as np


def get_stop_point(scores, threshold=0.01) -> int:
    """
    Returns the index of the first score that is below the threshold
    @param scores: NDS scores
    @param threshold: Threshold for stopping
    @return: Returns the index of the first score that is below the threshold
    """
    scores = np.concatenate(([np.min(scores[:2])], scores[2:]))
    diffs = scores[:-1] - scores[1:]
    for i, diff in zip(np.arange(1, len(scores) + 1), diffs):
        if diff < threshold:
            return i
