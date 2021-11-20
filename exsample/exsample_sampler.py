import time
import torch
import numpy as np
import pyroaring as pr

class Sampler:
    def next_batch(self,n):
        pass
    def feedback(self, *args):
        pass


class RandomlyPermutedRange(Sampler):
    def __init__(self, start, end, excluded_set=pr.FrozenBitMap()):
        super().__init__()
        assert end >= start
        self.start = start
        self.end = end
        self.remaining = pr.BitMap([])
        self.remaining.add_range(start, end)
        self.excluded_len = len(excluded_set)
        self.remaining.difference_update(excluded_set)

    def nf(self):
        return self.end - self.start - len(self.remaining) - self.excluded_len

    def next_batch(self, n):
        n = min(n,len(self.remaining))
        batch = np.zeros(n, dtype=np.int)
        for i in range(n):
            idx = np.random.randint(low=0, high=len(self.remaining))
            frm = self.remaining[idx]
            batch[i] = frm
            self.remaining.remove(frm)
        return (batch, None)

class PreOrder(Sampler):
    def __init__(self, ilocs, excluded_set=pr.BitMap([])):
        super().__init__()
        self.ilocs = ilocs
        self.i = 0
        self.excluded_set = excluded_set
        self.returned = 0

    def nf(self):
        return self.returned

    def next_batch(self, n):
        ret = np.ones(n, dtype=self.ilocs.dtype) * -1
        cr = 0
        while cr < n and self.i < self.ilocs.shape[0]:
            candidate = self.ilocs[self.i]
            self.i += 1
            if candidate not in self.excluded_set:
                ret[cr] = candidate
                cr += 1

        self.returned += cr
        return (ret[:cr], None)

def testPreOrder():
    excluded_set = pr.BitMap([])
    initial = 2 ** 10
    order = np.random.permutation(2 ** 20)[:initial]
    po = PreOrder(order, excluded_set=excluded_set)

    i = 0
    for n in range(10):
        ret, _ = po.next_batch(n=n)
        assert ret.dtype == order.dtype
        assert ret.shape[0] == n
        assert (ret == order[i:i + n]).all()
        i += n

    excluded_set.update(order[i:i + 100])
    ret, _ = po.next_batch(n=1)
    assert ret[0] == order[i + 100]
    assert ret[0] not in excluded_set

    excluded_set.add_range(0, 2 ** 20)
    ret, _ = po.next_batch(n=10)
    assert ret.shape[0] == 0
    ret, _ = po.next_batch(n=0)
    assert ret.shape[0] == 0

testPreOrder()

from .estimation import BanditFp,IncrementalFp
from collections import namedtuple, OrderedDict, deque
import array as A
import random
import pyroaring as pr
import pandas as pd

def rank_scores(score):
    """
    :param score: score array
    :return: returns ranking from highest to
    lowest score, while randomizing order of any score-ties
    """
    score = -score #  want ascending scores
    tiebreaker = np.random.randn(*score.shape)
    idx1 = np.argsort(tiebreaker)
    score = np.take_along_axis(score,idx1,axis=-1)
    idx2 = np.argsort(score, kind='stable')
    idx = np.take_along_axis(idx1,idx2,axis=-1)
    return idx

def test_ranking():
    ## test score ranking to ensure it is returning the highest scores first,
    tmp1 = (np.arange(100) * .1).reshape(10, 10)
    ranking = rank_scores(tmp1)
    assert (ranking == np.arange(10)[::-1]).all()

    # check ties are handled ok
    # check -inf is handled ok
    tmp = np.concatenate([np.ones(20), np.zeros(20), -np.ones(1)*np.inf])
    scores = np.stack([tmp, tmp])
    ranking = rank_scores(scores)
    assert (ranking[:, :20] <= 20).all()
    assert (ranking[:, 20:] >= 20).all()
    assert (ranking[:,-1] == 40).all()
    assert not (ranking[0] == ranking[1]).all()


class CompositeSampler(Sampler):
    def __init__(self, samplers, NI,
                 score_method=None,
                 score_opts={}):
        """ uses a shared stat """
        super().__init__()
        self.samplers = np.array(samplers)
        self.bfp = BanditFp(NI, self.samplers.shape[0])
        assert callable(score_method)
        # make new instance here.
        self.score_model = score_method(sampler=self, **score_opts)
        self.finished = torch.zeros(self.samplers.shape[0], dtype=torch.bool) # boolean

        self.count = 0
        self.history = OrderedDict([('iloc', A.array('l')),
                                    ('choice', A.array('i')),
                                   ])
        self.iloc_history = self.history['iloc']
        self.choice_history = self.history['choice']

        self.pending_feedback = pr.BitMap([])
        self.score_opts = score_opts
        self._check_rep()

    def get_hits(self):
        return self.Hs

    def _check_rep(self):
        if random.random() < 0.001 or self.count < 10:
            assert len(self.choice_history) == len(self.iloc_history)
            # assert len(self.score_history) == len(self.choice_history)
            # assert len(self.match_history) == len(self.choice_history) - len(self.pending_feedback)
            # assert len(self.choice_history) == self.count

            assert (self.bfp._total == self.count - len(self.pending_feedback))

    def has_next(self):
        return True

    def next_batch(self, n):
        self._check_rep()
        final_samples = []
        scores = self.score_model.score(batch_size=n)
        scores[:, self.finished] = -np.inf

        ranks = rank_scores(scores.numpy())
        for row in ranks:
            for idx in row:
                sampler = self.samplers[idx]
                (ilocs,_) = sampler.next_batch(1)
                if len(ilocs) > 0:
                    final_samples.append((idx,ilocs[0]))
                    break
                else:
                    self.finished[idx] = 1

        assert len(final_samples) <= n
        if len(final_samples) < n:
            assert self.finished.all()

        ilocs = []
        for (choice, iloc) in final_samples:
            self.choice_history.append(choice)
            self.iloc_history.append(iloc)
            ilocs.append(iloc)

        sample_ids = list(range(self.count, self.count + len(ilocs)))
        self.pending_feedback.add_range(self.count, self.count+len(ilocs))
        self.count += len(ilocs)
        self._check_rep()

        assert len(sample_ids) == len(ilocs)
        return (np.array(ilocs, dtype=np.int), np.array(sample_ids, dtype=np.int))

    def feedback(self, batch, match_ids):
        (ilocs, ids) = batch
        assert len(ids) == len(ilocs)
        assert len(ids) == len(match_ids)

        for (sample_id, match_id) in zip(ids, match_ids):
            self.pending_feedback.remove(sample_id) # will also check it is there
            # self.match_history.append(match_id)

            part_id = self.choice_history[sample_id]
            self.bfp.add(match_id, part_id)

    def get_sampler_history(self):
        return pd.DataFrame(self.history)

import torch
import torch.distributions as dist
import random
import scipy


class ScoringModel(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.stats = sampler.bfp

    def score(self, batch_size : int) -> torch.Tensor:
        raise NotImplementedError('must implement this')

class RandomScore(ScoringModel):
    def __init__(self, **base_args):
        super().__init__(**base_args)

    def score(self, batch_size):
        return dist.Normal(torch.zeros(self.stats.Nonce.shape[0]),
                           torch.ones(self.stats.Nonce.shape[0])).sample((batch_size,))


class GtScore(ScoringModel):
    def __init__(self, sampler, alpha_0=.1, beta_0=1.):
        super().__init__(sampler)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def score(self, batch_size):
        ds = dist.Gamma(concentration=self.stats.Nonce + self.alpha_0,
                        rate=self.stats.Ns + self.beta_0)
        s = ds.sample((batch_size,))
        return s

import scipy.stats

class StatScorer(ScoringModel):
    def __init__(self, sampler, *, stat='gt', exp_model='ts', alpha_0=.1, beta_0=1., ucb_quantile=.9):
        super().__init__(sampler)
        assert stat in ['gt', 'unique', 'uniform']
        assert exp_model in ['ts', 'bayes_ucb', 'none']
        self.stat = stat
        self.exp_model = exp_model
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.ucb_quantile = ucb_quantile

    def score(self, batch_size):
        if self.stat == 'gt':
            alpha = self.stats.Nonce + self.alpha_0
            beta = (self.stats.Ns + self.beta_0)
        elif self.stat == 'unique':
            alpha = self.stats.Nunique + self.alpha_0
            beta = (self.stats.Ns + self.beta_0)
        elif self.stat == 'hits':
            alpha = self.stats.Is + self.alpha_0
            beta = (self.stats.Ns + self.beta_0)
        elif self.stat == 'uniform':
            alpha = torch.ones_like(self.stats.Nunique)
            beta = torch.ones_like(self.stats.Nunique)
        else:
            assert False

        gg = scipy.stats.gamma(a=alpha.numpy(), scale=1 / beta.numpy())

        if self.exp_model == 'ts':
            score_vals = gg.rvs(size=(batch_size, beta.shape[0]))
        elif self.exp_model == 'bayes_ucb':
            n = self.stats.Ns.sum()
            quantile = 1 - 1. / (n + 1.)  # see paper. quantile increases with time for bayesucb
            sc = gg.isf(1. - quantile)  # inverse function of 1 - CDF
            score_vals = np.stack([sc for _ in range(batch_size)])
        elif self.exp_model == 'none':
            sc = gg.mean()
            score_vals = np.stack([sc for _ in range(batch_size)])
        else:
            assert False
        return torch.from_numpy(score_vals)


class UniqueScore(ScoringModel):
    def __init__(self, **base_args):
        super().__init__(**base_args)

    def score(self, batch_size):
        ds = dist.Gamma(concentration=self.stats.Nunique + .1, rate=self.stats.Ns + 1.)
        s = ds.sample((batch_size,))
        return s

import pyroaring as pr
import pandas as pd
import numpy as np

def score_minus(sc, ds,
                object_class,
                excluded_frames,
                initial_distance=32):
    scores = sc[object_class]
    hits = ds.ok_boxes[ds.ok_boxes.category == object_class].index
    assert (sc.index == pd.RangeIndex(ds.video.len)).all(), 'work with absolute index'

    excluded = pr.FrozenBitMap(excluded_frames)
    all_frames = pr.FrozenBitMap(sc.index)
    assert excluded.issubset(all_frames)

    acc = []
    seen = pr.FrozenBitMap([])
    dist = initial_distance
    while len(seen) < len(all_frames - excluded):
        locs = score_minus_helper(scores, hits,
                                  initial_ex=excluded.union(seen),
                                  distance=dist)
        dist = dist // 2
        seen = seen.union(pr.FrozenBitMap(locs))
        acc.append(locs)

    final = np.concatenate(acc)
    assert len(pr.BitMap(final)) == final.shape[0]  # no repeats
    assert pr.BitMap(final).intersection_cardinality(excluded) == 0  # no training set
    assert final.shape[0] == len(all_frames) - len(excluded)  # all remaining
    return pd.Index(final, name='blz_' + object_class)

def score_minus_helper(scores, hits, initial_ex=None, distance=32):
    banned = pr.BitMap(initial_ex)

    start_locs = pr.BitMap(scores.index).difference(banned)
    scores = scores.loc[start_locs]
    locs = scores.sort_values(ascending=False).index
    hits = pr.BitMap(hits)
    output_locs = []
    for l in locs:
        if l in banned:
            continue
        else:
            output_locs.append(l)
            #            if l in hits: # only clear range for positive result
            banned.add_range(l - distance, l + distance)

    return np.array(output_locs)