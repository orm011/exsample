from .exsample_benchmark import BenchmarkDataset
from collections import OrderedDict

def getPs(ds : BenchmarkDataset, dsplit):
    tmp = dsplit.value_counts().sort_index()
    split_ids = tmp.index
    split_sizes = tmp.values
    okb = ds.ok_boxes.assign(split=dsplit[ds.ok_boxes.frame_id].values)
    tns = okb.groupby(['track_id', 'category', 'split']).agg(
        track_length=pd.NamedAgg('frame_id', 'count')).reset_index()
    tns = tns.assign(split_size=split_sizes[tns.split])
    tns = tns.assign(p=tns.track_length / tns.split_size)
    pdf = tns[['category', 'split', 'p', 'track_id']]

    mats = []
    for (k, gp) in pdf.groupby('category'):
        mat = gp.pivot(values='p', columns='split', index='track_id').fillna(0)
        mat = mat.transpose().reindex(split_ids).fillna(0).transpose()
        assert mat.shape[1] == split_sizes.shape[0]
        mats.append((k, mat))

    return OrderedDict(mats)

def get_histogram(ds, split):
    ps = []
    pss = getPs(ds, split)
    for (k,pdf) in pss.items():
        N = (pdf > 0).sum(axis=0)
        mup =  pdf.sum(axis=0).values/N.values
        sigma = np.sqrt(((pdf**2).sum(axis=0).values - mup**2)/N.values)
        df = pd.DataFrame({'N':N, 'mup':mup, 'sigma':sigma})
        ps.append(df.assign(category=k, dataset=ds.name, M=N.shape[0], chunk=np.arange(df.shape[0])))
    return pd.concat(ps, ignore_index=True)

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def missing(P, wns, lib=np):
    return lib.sum(lib.exp(-(P @ wns)))

def log_missing(P, wns):
    return cp.log_sum_exp(-P @ wns)

def expected(ps,n):
    ps = ps.reshape(-1,1)
    n = n.reshape(1,-1)
    ans = ((1-ps)**(n)).sum(axis=0)
    return ps.shape[0] - ans

class MaxN(object):
    def __init__(self, P):
        """
        :param P: p matrix for dataset x chunk combination
        Computes optimal w for minimizing missing instances after \sum w tries.
        """
        self.P = P
        self.wns = cp.Variable(P.shape[1], )
        self.n = cp.Parameter()
        self.prob = cp.Problem(cp.Minimize(missing(P, self.wns,lib=cp)),
                               [self.wns >= 0,
                                cp.sum(self.wns) == self.n,
                                ])

    def solve(self, n, **optns):
        self.n.value = n
        loss_val = self.prob.solve(**optns)
        return self.P.shape[0] - loss_val


class OptW(object):
    def __init__(self, P):
        """
        :param P: p matrix for dataset x chunk combination
        Computes optimal w for minimizing missing instances after \sum w tries.
        """
        self.P = P
        wns = cp.Variable(P.shape[1], )
        self.wns = wns
        lam = cp.Parameter()
        self.lam = lam
        self.prob = cp.Problem(cp.Minimize(lam * cp.sum(wns) + missing(P, wns,lib=cp)),
                               [wns >= 0])

    def solve(self, lam, **optns):
        self.lam.value = lam
        loss_val = self.prob.solve(**optns)
        return self.wns.value, loss_val


class OptWn(object):
    def __init__(self, P):
        """
        :param P: p matrix for dataset x chunk combination
        Computes optimal w for minimizing missing instances after \sum w tries.
        """
        self.P = P

        wns = cp.Variable(P.shape[1], )
        self.n = cp.Parameter()
        self.prob = cp.Problem(cp.Minimize(missing(P, wns,lib=cp)),
                               [wns >= 0,
                                wns.sum() == self.n,
                                ])

    def solve(self, n, **optns):
        self.n.value = n
        loss_val = self.prob.solve(**optns)
        return self.P.shape[0]  - loss_val

class MatchN(object):
    def __init__(self, P, w):
        """
        :param P: p matrix for dataset x chunk combination
        :param w: a fraction of samples used for each partition
        Computes the n (number of samples) that will match expected misses.
        """
        assert np.isclose(w.sum(), 1.)
        self.N = cp.Variable()
        self.match_loss = cp.Parameter()
        self.prob = cp.Problem(cp.Minimize(self.N),
                               [missing(P, self.N * w, lib=cp) <= self.match_loss]
                               )

    def solve(self, misses, **opts):
        self.match_loss.value = misses
        self.prob.solve(**opts)
        return self.N.value


class WeightSolver(object):
    def __init__(self, P):
        self.P = P
        unif = np.ones(P.shape[1]) / P.shape[1]
        self.optw = OptW(P)
        self.matchN = MatchN(P, unif)
        self.warm_start = False

    def _try_resolve(self, lam_val, **optns):
        P = self.P
        try:
            wns, loss = self.optw.solve(lam=lam_val, **optns)
        except cp.SolverError as e:
            print('failed optw')
            raise e

        misses = missing(P, wns)
        actualN = P.shape[0] - misses
        nrandom = self.matchN.solve(misses=misses, solver='ECOS', max_iters=200, warm_start=True)
        return (lam_val, loss, actualN, nrandom, wns)

    def _resolve(self, lam_val):
        default_optns = dict(solver='ECOS', max_iters=300)

        try:
            sol = self._try_resolve(lam_val, warm_start=True, **default_optns)
            return sol
        except cp.SolverError:
            pass
        return None

    def resolve(self, lamvals):
        res = []
        for lm in tqdm(lamvals, desc='lambdas'):
            r = self._resolve(lm)
            if r is not None:
                res.append(r)

        if len(res) == 0:
            return None

        return pd.DataFrame(res, columns=['lam', 'loss', 'actualN', 'nrandom', 'wns'])

def estimate_lambdas(P):
    psum = P.sum(0)
    lams = np.quantile(psum[psum > 0], [.99, 0.])
    lams[-1] = lams[-1]/8
    return lams

def get_target_lambdas(P, lambdas=None, nlambdas=8):
    if lambdas is None:
        lambdas = estimate_lambdas(P)
        lambdas = np.geomspace(*lambdas, nlambdas)
        print('Using lambda range: ', lambdas)

    ws = WeightSolver(P)
    rs = ws.resolve(lambdas)
    if rs is None:
        return rs
    vecs = np.stack(rs.wns.values)
    optn = vecs.sum(axis=1)
    normws = vecs/optn.reshape(-1,1)
    unif = np.ones(P.shape[1])/P.shape[1]
    wdist = np.linalg.norm(normws-unif.reshape(1,-1),axis=1)
    rs = rs.assign(optn=optn, ratio=rs.nrandom/optn, wdist=wdist, recall=rs.actualN/P.shape[0])
    return rs