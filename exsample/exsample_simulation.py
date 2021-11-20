import numpy as np
import torch.distributions as dist
import torch
from .multimap import ArrayMultiMap


def mean_std(m, s):
    """
    :param m: mean of log(X), X~LogNormal
    :param s: std of log(X)
    :return: expectation and sqrt(var) for X
    """
    M = np.exp(m + (s**2)/2)
    S2 = (np.exp(s**2) - 1)*np.exp(2*m + s**2)
    return M, np.sqrt(S2)

def inv_mean_std(M, S):
    """
    :param M: mean of X X~LogNormal
    :param S: std. of X
    :return: mu, sigma for log(X), which are used as parameters eg. in torch.
    """
    rad = np.sqrt(M**2 + S**2)
    m = np.log(M**2/rad)
    s = np.sqrt(2*(np.log(rad/M)))
    return m, s

def generate_positions(N, NF, sigmas):
    pos = dist.Normal(sigmas * 4.5, sigmas).sample((N,))  # 95% of things in the +/- sigma centered at the mean
    pos = pos.round() % NF
    return pos

def generate_durations(N, NF, mu, sigma):
    prob = dist.LogNormal(loc=mu, scale=sigma).sample((N,))
    durations = (prob * NF).ceil()
    return durations

def generate_tracks(positions, durations, NF):
    durations = durations.view(-1, 1)
    starts = (positions - durations // 2).clamp(0, NF - 1).int()
    ends = (positions + (durations // 2) + (durations % 2)).clamp(0, NF - 1).int()
    assert (starts >= 0).all()
    assert (ends < NF).all()
    assert ((ends - starts) >= 1).all()
    return starts, ends

def make_multi_map(starts, ends):
    sz = (ends - starts).sum().item()
    frames = torch.zeros((sz,), dtype=torch.int)
    tracks = torch.zeros((sz,), dtype=torch.int)

    offset = 0
    for (i, (s, e)) in enumerate(zip(starts, ends)):
        l = e - s
        tracks[offset:offset + l] = i
        frames[offset:offset + l] = torch.arange(s, e)
        offset += l

    am = ArrayMultiMap(keys=frames.numpy(), values=tracks.numpy())
    return am

def getPs(am: ArrayMultiMap, N: int, NF: int, chsz: int):
    """
    :param am: multimap frames => track ids
    :param N: total instances
    :param NF: total frames in dataset
    :param chsz: chunk size (NF// chsz) = M
    :return:
    """
    assert NF % chsz == 0
    frms = am.okeys
    tids = am.ovalues
    splits = frms // chsz
    nsplits = NF // chsz
    unqs = tids * nsplits + splits
    locs, cts = np.unique(unqs, return_counts=True)
    row = locs // nsplits
    col = locs % nsplits
    Ps = np.zeros((N, nsplits))
    Ps[row, col] = cts
    Ps = Ps / chsz
    return Ps


def make_score_order(mm : ArrayMultiMap, N : int):
    """
    :param mm: frames => tids
    :param N: total number of instances
    :return: a score for each frame, and the ordering on frames
    based on a decreasing score
    """
    scores_pos = dist.Normal(.7, .25/2).sample((N,)).numpy()
    scores_neg = dist.Normal(.2, .25/2)
    frscore = scores_pos[mm.ovalues]
    np.unique(mm.okeys).shape
    unq_keys, cts = np.unique(mm.okeys, return_counts=True)
    accs = frscore.cumsum()
    tots = cts.cumsum()
    lastaccs = accs[tots-1]
    frame_total = np.diff(lastaccs, prepend=0)
    score_order = np.argsort(-frame_total)
    fids = unq_keys[score_order]
    scores = frame_total[score_order]
    return fids, scores

# param values
# NF = int(2**24) # 16 million
# N = 2000
# lmu, lsigma = inv_mean_std(M=700/NF, S=500/NF)
#
# pows = torch.arange(1, 13, step=3).float()
# Ms = 2.**pows # num partitions: 2 to 1024.
# chunk_size = NF/Ms
#
