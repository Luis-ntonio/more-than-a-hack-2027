"""Viterbi decoding of the H / A / D chain."""
import numpy as np

STATES = np.array(["H", "A", "D"])


def viterbi(log_emission, log_transition, log_start=None):
    """Most likely state sequence.

    log_emission: (n, k) per-frame log-likelihoods. log_transition: (k, k), row = from state.
    Runs in O(n k^2) with a Python loop over frames; ~80k frames x 3 states takes about a second.
    """
    n, k = log_emission.shape
    score = (np.zeros(k) if log_start is None else log_start) + log_emission[0]
    back = np.zeros((n, k), dtype=np.int8)
    for t in range(1, n):
        candidates = score[:, None] + log_transition  # (from, to)
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(k)] + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
