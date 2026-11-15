"""Decoding with possession lengths: a semi-Markov chain approximated by stages.

The plain HMM gives every state a geometric length: the shortest stay is always the likeliest.
Possessions are not like that, and spell F1 scores whole possessions. Here each of H, A and D is a
chain of k stages, each left with the same probability q, so its length follows a negative binomial
(k = 1 is the plain HMM). k and q per state come by maximum likelihood from the training labels'
run lengths, never from the provided match; what follows a state comes from the same counts as
external.train.transition_costs. Viterbi then runs on the expanded chain and stages are merged back.
"""
import numpy as np
from scipy.special import gammaln

from possession.decode import viterbi

STATES = np.array(["H", "A", "D"])
MAX_STAGES = 8


def run_lengths(labels):
    """{state: lengths in frames} of the runs in one label sequence."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    out = {s: [] for s in STATES}
    for a, b in zip(starts, ends):
        out[labels[a]].append(b - a)
    return out


def fit_stages(lengths):
    """(k, q) maximising the likelihood of these lengths: k stages, each left with probability q."""
    L = np.asarray(lengths, dtype=float)
    best = None
    for k in range(1, MAX_STAGES + 1):
        if L.min() < k:
            break
        q = k / L.mean()
        ll = np.sum(gammaln(L) - gammaln(k) - gammaln(L - k + 1) + k * np.log(q) + (L - k) * np.log1p(-q))
        if best is None or ll > best[0]:
            best = (ll, k, q)
    return best[1], best[2]


def fit(all_labels):
    """Duration model from training label sequences: {state: (k, q)} and where each state goes next."""
    lengths = {s: [] for s in STATES}
    nxt = np.ones((3, 3))
    index = {s: i for i, s in enumerate(STATES)}
    for labels in all_labels:
        for s, ls in run_lengths(labels).items():
            lengths[s] += ls
        change = np.flatnonzero(labels[1:] != labels[:-1])
        np.add.at(nxt, ([index[x] for x in labels[change]], [index[x] for x in labels[change + 1]]), 1)
    np.fill_diagonal(nxt, 0)
    return {"stages": {s: fit_stages(lengths[s]) for s in STATES}, "next": nxt / nxt.sum(axis=1, keepdims=True)}


def expanded(model):
    """(log transition over all stages, state index of each stage)."""
    ks = [model["stages"][s][0] for s in STATES]
    first = np.concatenate([[0], np.cumsum(ks)[:-1]])
    owner = np.repeat(np.arange(3), ks)
    m = sum(ks)
    T = np.zeros((m, m))
    for i, s in enumerate(STATES):
        k, q = model["stages"][s]
        for j in range(k):
            here = first[i] + j
            T[here, here] = 1 - q
            if j + 1 < k:
                T[here, here + 1] = q
            else:
                for o in range(3):
                    if o != i:
                        T[here, first[o]] = q * model["next"][i, o]
    with np.errstate(divide="ignore"):
        return np.log(T), owner


def decode(p_dead, p_home, half, model):
    """Like external.train.decode, on the expanded chain."""
    p_dead = np.clip(p_dead, 1e-6, 1 - 1e-6)
    p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
    emission = np.column_stack([np.log1p(-p_dead) + np.log(p_home),
                                np.log1p(-p_dead) + np.log1p(-p_home), np.log(p_dead)])
    log_t, owner = expanded(model)
    out = np.empty(len(half), dtype="<U1")
    for h in np.unique(half):
        rows = half == h
        out[rows] = STATES[owner[viterbi(emission[rows][:, owner], log_t)]]
    return out
