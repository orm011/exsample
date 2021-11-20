from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
from builtins import object
from collections import namedtuple, OrderedDict, Counter
import pyroaring as pr
import random
import torch
import pandas as pd
import numpy as np
import tqdm
if False:
    import cvxpy as cvx
from torch import distributions as dist
from functools import reduce
from .multimap import Sparse2DenseMapper

## change BanditFp so that the match_id can be an arbitrary int,
## which we translate automatically...
## id -> dense_id

class BanditFp(object):
    def __init__(self, NI, num_parts):
        k_init = 3
        self._total = 0
        self.id_mapper = Sparse2DenseMapper(2**21)
        self.c1 = torch.zeros(NI+1, num_parts, dtype=torch.int32)
        self.c2 = torch.zeros(k_init, num_parts)

        self.gfp = None
        if num_parts > 1:
            self.gfp = BanditFp(NI, num_parts=1)

        self.Hs = torch.zeros(num_parts) # binary: frames with something in them
        self.Is = torch.zeros(num_parts) # number of instances (non-dedup)
        self.Ns = torch.zeros(num_parts)

        self.pos_count = None
        self.Nonce = None
        self.Nonceglobal = torch.zeros(num_parts)
        self.part_ids = torch.arange(num_parts).int()

        self._refresh_c2()
        self.pc = 0.5
        self._check_rep()

    def _refresh_c2(self):
        self.c2 = torch.cat([self.c2, torch.zeros_like(self.c2)])
        self.pos_count = torch.arange(self.c2.shape[0]).float()
        self.Nonce = self.c2[1, :]

    def _refresh_gc2(self):
        self.gc2 = torch.cat([self.gc2, torch.zeros_like(self.gc2)])

    def _check_rep(self):
        r = random.random()
        frac = 1./5000
        lim = 3

        if r < frac or self._total < lim:
            #assert ((self.Hs.int() + self.Ms.int()).int() == self.Ns.int()).all()
            assert self._total == self.Ns.sum()


    def add(self, match_ids, part_id):
        self._check_rep()
        self._total += 1
        self.Ns[part_id] += 1
        self.Is[part_id] += len(match_ids)
        self.Hs[part_id] += (len(match_ids) > 0)

        curr = -1
        for match_id in match_ids:
            match_id = self.id_mapper[match_id]
            entry = self.c1[match_id, part_id]
            entry += 1 # increment
            curr = entry.item()

            if curr >= self.c2.shape[0]:
                self._refresh_c2()

            self.c2[curr - 1, part_id] -= 1
            self.c2[curr, part_id] += 1

        self._check_rep()
        return curr

    @property
    def Nunique(self):
        return self.c2[0, :].abs()


class IncrementalFp(object):
    def __init__(self):
        self.instances_seen = pr.BitMap() ## seen at all
        self.instances_seen_once = pr.BitMap()
        self.c1 = Counter()
        init_len = 5
        self.c2arr = np.zeros(init_len, dtype=np.float)
        self.ks = np.arange(init_len)
        assert self.c2arr.shape[0] == self.ks.shape[0]
        self.delta = np.array([-1.,1.])
        self.N = 0



    def _check_rep(self):
        if self.N in [0, 1, 2, 10,11, 100,101] or random.random() <= 0.001:
            assert len(self.instances_seen) == -self.c2arr[0]
            assert len(list(self.c1.keys())) == len(self.instances_seen)
            assert len(self.instances_seen_once) == self.c2arr[1]
            assert sum(self.c1.values()) == self.N
            assert len(list(self.c1.keys())) == self.ni

            assert self.instances_seen_once.issubset(self.instances_seen)

            nparr =  self._get_np()
            k = nparr.shape[0]
            assert self.c2arr.shape[0] >= k #nparr.shape[0]
            assert (self.c2arr[1:k] == nparr[1:]).all()
            assert (self.c2arr[k:] == 0).all()

            assert self.ks.dot(self.c2arr) == self.N

    def add(self, elt):
        self._check_rep()

        self.N +=1
        self.c1.update([elt])
        ct = self.c1[elt]

        if ct == 1:
            assert elt not in self.instances_seen
            self.instances_seen.add(elt)
            self.instances_seen_once.add(elt)

        if ct == 2:
            assert elt in self.instances_seen_once
            self.instances_seen_once.remove(elt)

        if ct >= self.c2arr.shape[0]:
            # realloc
            old = self.c2arr
            k = old.shape[0]
            self.c2arr = np.zeros(2*k)
            self.c2arr[:k] = old
            self.ks = np.arange(2*k)

        self.c2arr[ct-1:ct+1] += self.delta
        self._check_rep()
        return ct == 1 ## indicate if new result

    @property
    def ni(self):
        return -self.c2arr[0]

    @property
    def n1(self):
        return self.c2arr[1]


    def good_turing_estimate(self):
        if self.N == 0:
            return 1.0

        return self.n1/1.0/self.N

    def good_toulmin_estimate(self, nsteps):
        if self.N == 0:
            return 1.0*nsteps

        t = - (nsteps / 1.0 / self.N)
        tpow = np.power(t, self.ks)
        return - (self.c2arr[1:].dot(tpow[1:]))

    def get_np(self):
        ans = self.c2arr.copy()
        ans[0] = 0
        return ans

    def _get_np(self):
        c2 = Counter(list(self.c1.values()))
        mx = max(c2.keys()) if len(c2) > 0 else 1
        assert mx > 0
        emp = np.zeros(mx + 1)

        for (k, v) in list(c2.items()):
            emp[k] = v

        return emp

    def get_torch(self):
        return torch.tensor(self.get_np())



Fingerprint = namedtuple('Fingerprint', 'h lx')


def get_dist(match_series, split_series, normalize=True):
    assert (match_series.shape == split_series.shape)
    df2 = pd.DataFrame({'inst': pd.Categorical(match_series).codes, 'split': pd.Categorical(split_series).codes})
    ans = df2.pivot_table(index='inst', columns='split', aggfunc='size', fill_value=0)
    assert ans.values.sum() == match_series.shape[0], "all frames need to be accounted for"
    if normalize:
        ans = ans / 1. / ans.sum()

    return ans

def get_torch_dist(instances=None, splits=None,
                   instance_codes=None, split_codes=None, normalize=True, also_pd=False, totals=False):

    if instance_codes is not None:
        instances = instance_codes
    if split_codes is not None:
        splits = split_codes

    npdist_raw = get_dist(instances, splits, normalize=False)
    npdist_totals = npdist_raw.sum()

    npdist = npdist_raw/1.0/npdist_totals
    out = torch.tensor(npdist.values).float()

    ans = out

    if also_pd:
        ans = [out, npdist]

    if totals:
        ans = [out, npdist, torch.tensor(npdist_totals.values).float()]

    return ans


def soft_fingerprint(dist, grid=None, step_factor=1.3):
    """
        given a grid, creates a soft mapping of the dist params
        into the grid.
        Unlike methods using histograms,
        we can back-propagate from the output h to the input dist.
        (lgrid is considered given)
    """
    if grid is None:
        lgrid_min = dist.min().log().detach()
        lgrid_step = torch.tensor(step_factor).log()
        step_fl = (dist.log().detach() - lgrid_min).div(lgrid_step).round()
        M = step_fl.max().detach()
        lgrid = (lgrid_min + torch.arange(0, M) * lgrid_step)
    else:
        lgrid = grid.log()

    lstep = (lgrid[1:] - lgrid[:-1]).min()

    scores = (dist.log().view(-1, 1) - lgrid.view(1, -1)).div(lstep).abs().neg()
    # want one full step to account for a 10x diff in probability
    temp = np.log(10.)
    h = scores.mul(temp).softmax(dim=-1).sum(dim=0)
    return Fingerprint(h=h, lx=lgrid)

def get_fingerprint(dist, grid=None, step_factor=1.2):
    ## not sure what happens if we have multiple distributions.
    dist = dist[dist > 0] # b/c of partitioning at a higher level, we may get zero entries that don't affect the hst.
    if grid is None:
        lgrid_min = dist.min().log().detach()
        lgrid_step = torch.tensor(step_factor).log()
        step_fl = (dist.log().detach() - lgrid_min).div(lgrid_step).round()
        M = step_fl.max().detach()
        lgrid = (lgrid_min + torch.arange(0, M) * lgrid_step)
        h = step_fl.histc(bins=M.int(), max=M.detach().item(), min=0)
    elif grid == 'no':
        pass

    return Fingerprint(h=h, lx=lgrid)


def expected_reward(dists, steps=1, poisson_approx=False, lsb=False):
    dists = torch.tensor(dists).double()
    steps = torch.tensor(steps).double()

    if ((dists.sum(dim=0)  -  1.).abs() < 0.01).all():
        print('all partitions have 0 prob of a miss. are we counting misses as a kind of instance?')
        assert False

    sz = torch.Size([1] * len(dists.shape) + [-1])
    dists = dists.unsqueeze(-1) # add dim for steps
    steps = steps.view(sz) # add dime for dists.
    non_misses = dists  ## should have removed params not related to misses

    if lsb:
        # min(kp, 1)
        rates = non_misses * steps  # zero rates yield nan... (bug?) want 0.

        p_hit = 1. - (1. - rates).relu()

    if not lsb:
        if not poisson_approx:
            # 1 - (1-p)^k
            lp_miss_one = (1. - non_misses).log()
            lp_miss_k = lp_miss_one * steps
            p_miss_k = lp_miss_k.exp()
            p_hit = 1. - p_miss_k
        else:
            # random: 1 - exp(kp)
            rates = non_misses * steps  # zero rates yield nan... (bug?) want 0.

            lp_fail = -rates  # poisson(lambda, 0)  = exp(-lambda) => logprob is -lambda
            p_hit = -lp_fail.expm1()

    return p_hit.sum(dim=0)


def expected_reward3(fingerprint, steps=1, loss=False):
    steps = torch.tensor(steps).float()
    if steps.ndimension() == 1:
        steps = steps.view(1, -1)

    pfail = torch.distributions.Poisson(rate=fingerprint.lx.exp().view(-1, 1) * steps).log_prob(0).exp()
    return fingerprint.h.sum() - pfail.t().mv(fingerprint.h)



def get_expected(matches, lsb=False):
    # for no partitioning used.
    expected = []
    assume_size = matches.shape[0]

    fractions = np.power(10, np.arange(-4.000, -0.99, step=0.125))# 1/1000 to 1/10th in increments of 3.3x. 0.001, 0.0033.
    steps = np.floor(assume_size*fractions).astype('int64')
    steps = np.sort(np.concatenate([steps, steps+1]))
    for c in matches.columns:
        dist_c = get_torch_dist(matches[c])
        er_c = expected_reward(dist_c[1:], steps=steps, lsb=lsb)
        df = pd.DataFrame(OrderedDict([('n_ground_matched', er_c), ('frames_inspected', steps)]))
        df['object_class'] = c
        df['num_ground'] = dist_c[1:].shape[0]
        df['sampler_name'] = 'expected_random' if not lsb else 'expected_lsb'
        df['sample_fraction'] = df.frames_inspected/1.0/assume_size
        df['num_frames'] = assume_size
        df['recall'] = df.n_ground_matched/1.0/df.num_ground
        df['sampler_id'] = pd.Timestamp.now()
        expected.append(df)

    cdf = pd.concat(expected, sort=True, ignore_index=True)
    cdf['runs'] = 1
    return cdf


import matplotlib.pyplot as plt
def width_plot(instances, relative_y=False, relative_x=False, log_x=True, log_y=False,
               rev_cdf=False):
    bicycle_dist = get_dist(instances)
    bicycle_fp = bicycle_dist[0].loc[1:].value_counts().sort_index()

    h = bicycle_fp.values
    x = bicycle_fp.index
    # bicycle_fp = get_fingerprint(bicycle_dist[1:], step_factor=2.)
    # mass1 = bicycle_dist[1:].sum()
    # mass2 = bicycle_fp.h.dot(bicycle_fp.lx.exp())
    # print mass1, mass2, mass1 - mass2, (mass1 - mass2) / mass1
    ninstances = h.sum()
    #bicycle_fp.h.sum()
    #lx = bicycle_fp.lx[bicycle_fp.h > 0]
    #h = bicycle_fp.h[bicycle_fp.h > 0]

    if rev_cdf:
        h = ninstances - (h.cumsum())
        a = np.zeros_like(h)
        a[1:] = h[0:-1]
        a[0] = ninstances
        h = a

    if relative_y:
        h = h/1.0/ninstances

    if relative_x:
        x = x/1.0/instances.shape[0]

    if log_x and log_y:
        func = plt.loglog
    elif log_x:
        func = plt.semilogx
    elif log_y:
        func = plt.semilogy
    else:
        func = plt.plot

    func(x, h, alpha=0.5, marker='.', label=instances.name)
    plt.xlabel('# frames with instance' if not relative_x else 'probability of instance')

    ylabel = '# instances' if not relative_y else 'fraction of instances'
    if rev_cdf:
        ylabel = 'cumulative ' + ylabel
    plt.ylabel(ylabel)
    plt.legend()


def oom_plot(res, **kwargs):
    plt.figure()
    for c in res:
        width_plot(res[c], **kwargs)


def expected_reward_cvx(tdist, palloc, future_steps,
                        poisson_approx=True, # works more stable
                        proxy_reward=False, # use -log_sum_exp rather than actual rw.
                        lsb=False):
    assert ((tdist.sum(dim=0) - 1.).abs() < 1e-5).all()

    ## remove non-answer
    tdist = tdist[1:]
    N = tdist.shape[0]

    # sure_shots = (tdist.sum(dim=0) == 1.).sum()
    # # if sure_shots > 0:
    # #     print('partition has ', sure_shots)

    mdist = cvx.matmul(tdist.numpy(), palloc)
    future_steps = float(future_steps)

    if not lsb:
        if poisson_approx:
            rates = mdist * future_steps
            return N - cvx.sum(cvx.exp(-rates))
        else: ## only works for verification
            fail_rate = (1. - mdist.value) ## cvx.power fails with some strange error
            fail_all = np.power(fail_rate, future_steps)
            return N - np.sum(fail_all)
    else:
        rates = mdist * future_steps
        min1x =  - cvx.neg(rates - 1.)
        return N - cvx.sum(min1x)



def optimal_allocation_cvx(tdists, future_steps, cvx_opts, poisson_approx=True, **kwopts):
    palloc = cvx.Variable(tdists.shape[1])
    reward = expected_reward_cvx(tdists, palloc, future_steps, poisson_approx=poisson_approx, **kwopts)
    problem = cvx.Problem(cvx.Maximize(reward), constraints=[cvx.sum(palloc) == 1., palloc >= 0])
    cvx_opts2 = cvx_opts.copy()

    if 'solver' not in cvx_opts:
        cvx_opts2['solver'] = ['ECOS']
    if type(cvx_opts['solver']) is str:
        cvx_opts2['solver'] = [cvx_opts['solver']]

    solvers = cvx_opts['solver']
    rw = None
    print(('using solvers', solvers))
    for sl in solvers:
        cvx_opts2['solver'] = sl
        try:
            rw = problem.solve(**cvx_opts2)
            if problem.status == 'optimal':
                break
            else:
                print('Problem status not optimal: %s' % problem.status)
                print('trying another solver')
        except cvx.SolverError:
            print('solver {} failed.'.format(sl))

    if rw is None:
        print(('all solvers failed', solvers))
        x = np.ones(tdists.shape[1])
        return 0., 0., old_div(x,x.sum()), problem, sl
    elif problem.status != 'optimal':
        print(('Warning: problem status: %s', problem.status))

    ## get reward value without using approx
    achieved_rw = expected_reward_cvx(tdists, palloc.value, future_steps, poisson_approx=False, **kwopts)
    return rw, achieved_rw, palloc.value, problem, sl

def optimal_allocation(tdists, future_steps, max_iter=int(1e5), verbose=False, lsb=False):
    alloc = torch.zeros(tdists.shape[1], requires_grad=True)
    opt = torch.optim.Adam(lr=0.004, params=[alloc])

    best_alloc_so_far = None
    early_stop_th = 4000  # will stop if we havent improved on our best reward by 1 in the last N steps.

    best_so_far = [0.]
    for i in range(max_iter):
        opt.zero_grad()
        palloc = alloc.softmax(dim=-1)
        mixed_dist = tdists.mv(palloc)
        er = expected_reward(mixed_dist[1:], steps=future_steps, lsb=lsb)

        if er > best_so_far[-1]:
            best_so_far.append(er.detach().item())
            best_alloc_so_far = torch.tensor(alloc.detach())
            best_it = i
        else:
            best_so_far.append(best_so_far[-1])

        if verbose and i % 1000 == 0:
            print(i, ' er ', er.detach().item(), ' best_er ', best_so_far[-1])

        if len(best_so_far) > early_stop_th:
            curr = best_so_far[-1]
            past = best_so_far[-early_stop_th]

            if curr - past < 1.:
                print('early stopping at i=%d: curr: %.2f  past: %.2f' % (i, curr, past))
                break

        loss = -er
        loss.backward()
        opt.step()

    achieved_dist = tdists.mv(best_alloc_so_far.softmax(-1))[1:]
    assert expected_reward(achieved_dist, future_steps, lsb=lsb) == best_so_far[-1]
    return best_so_far[-1], best_alloc_so_far

def label_alloc(df, pdist, nsamples):
    l1 = df.opt_alloc.apply(pd.Series)
    l2 = l1.rename(axis=1, mapper=lambda x: pdist.columns[x])
    l3 = l2.rename(axis=0, mapper=lambda x: nsamples[x])
    return l3


def opt_alloc_results(matches_col, splits_col,
                      n_steps=[10, 30, 100, 300, 600, 1000, 1800,3000, 6000, 10000, 18000, 30000],
                      lsb=False, cvx_opts=dict(solver='ECOS')):

    assert matches_col.name is not None
    if type(splits_col) is not list:
        splits_col = [splits_col]

    names = [x.name for x in splits_col]
    full_name = '_'.join(names)
    dists = []
    pdists = []
    split_names = []
    for c in splits_col:
        cc = pd.Categorical(c)
        split_names.append(list(cc.categories))
        tdist, pdist = get_torch_dist(matches_col, c, also_pd=True)
        dists.append(tdist)
        pdists.append(pdist)


    insts = pd.Categorical(matches_col).categories
    cnames = reduce(lambda a,b:a+b, split_names, [])
    tdists = torch.cat(dists, dim=-1)

    pdist =  pd.concat(pdists, axis=1)
    pdist = pdist.rename(axis=1, mapper=lambda x: np.NaN if x == -1 else cnames[x])
    pdist = pdist.rename(axis=0, mapper=lambda x: np.NaN if x == -1 else insts[x])


    best_score = []
    best_alloc = []
    status = []
    solver = []
    for steps in n_steps:
        sc, actualsc, alloc, prob, sl = optimal_allocation_cvx(tdists, future_steps=steps, lsb=lsb, cvx_opts=cvx_opts)
        best_score.append(actualsc)
        best_alloc.append(alloc)
        status.append(prob.status)
        solver.append(sl)

    ideal_alloc = pd.DataFrame(OrderedDict([('n_ground_matched', best_score),
                                            ('frames_inspected', n_steps),
                                            ('status', status),
                                            ('solver', solver),
                                            ('opt_alloc', best_alloc),
                                            ]))

    ideal_alloc = ideal_alloc.assign(sampler_name='opt_alloc_%s_%s' % (full_name, 'random' if not lsb else 'lsb'),
                                     object_class=matches_col.name,
                                     num_frames=matches_col.shape[0],
                                     runs=1,
                                     num_ground=len(matches_col.unique()))

    return ideal_alloc, pdist, label_alloc(ideal_alloc,  pdist, n_steps)

def expected_results(matches_col, n_steps = [10, 30, 100, 300, 600, 1000, 1800,
                                 3000, 6000, 10000, 18000, 30000, 60000, 100000]):
    tdist = get_torch_dist(matches_col)
    er = expected_reward(tdist, steps=n_steps)
    return pd.Series()


def test_split(split, gt, lsb=False, cvx_opts={}):
    opt_res = []
    for c in tqdm.tqdm_notebook(gt.columns):
        try:
            res = opt_alloc_results(gt[c], split, lsb=lsb, cvx_opts=cvx_opts)
            failed = False
        except:
            failed = True

        if failed:
            print(('failed on', c))
            # res = opt_alloc_results(gt[c], split, verbose=verbose, lsb=lsb, solver_opts=solver_opts)

    r = pd.concat(opt_res, sort=True)
    r['sample_fraction'] = old_div(r.frames_inspected,r.num_frames)
    r['recall'] = old_div(r.n_ground_matched,r.num_ground)

    return r


def easy_alloc(alloc, ores=None):
    ncols = len(alloc.columns)
    eps = 1. / ncols / 2.
    gd = alloc > eps
    any_gd = gd.any(axis=0).values
    gd_cols = alloc.columns[any_gd]
    r = alloc[gd_cols]

    if ores is not None:
        return r.set_index([r.index, ores.n_ground_matched.round(1)])


# def get_track_ps(tracks_df, split_series):
def expected_reward_cvx2(tdist, palloc, future_steps,
                         poisson_approx=True,  # works more stable
                         proxy_reward=False,  # use -log_sum_exp rather than actual rw.
                         lsb=False):
    # assert ((tdist.sum(dim=0) - 1.).abs() < 1e-5).all()

    ## remove non-answer
    # tdist = tdist[1:]
    N = tdist.shape[0]

    # sure_shots = (tdist.sum(dim=0) == 1.).sum()
    # # if sure_shots > 0:
    # #     print('partition has ', sure_shots)

    mdist = cvx.matmul(tdist.numpy(), palloc)
    future_steps = float(future_steps)

    if not lsb:
        if poisson_approx:
            rates = mdist * future_steps
            return N - cvx.sum(cvx.exp(-rates))
        else:  ## only works for verification
            fail_rate = (1. - mdist.value)  ## cvx.power fails with some strange error
            fail_all = np.power(fail_rate, future_steps)
            return N - np.sum(fail_all)
    else:
        rates = mdist * future_steps
        min1x = - cvx.neg(rates - 1.)
        return N - cvx.sum(min1x)


def optimal_allocation_cvx2(tdists, future_steps, cvx_opts, poisson_approx=True, **kwopts):
    palloc = cvx.Variable(tdists.shape[1])
    reward = expected_reward_cvx2(tdists, palloc, future_steps, poisson_approx=poisson_approx, **kwopts)
    problem = cvx.Problem(cvx.Maximize(reward), constraints=[cvx.sum(palloc) == 1., palloc >= 0])
    cvx_opts2 = cvx_opts.copy()

    if 'solver' not in cvx_opts:
        cvx_opts2['solver'] = ['ECOS']
    if type(cvx_opts['solver']) is str:
        cvx_opts2['solver'] = [cvx_opts['solver']]

    solvers = cvx_opts['solver']
    rw = None
    print(('using solvers', solvers))
    for sl in solvers:
        cvx_opts2['solver'] = sl
        try:
            rw = problem.solve(**cvx_opts2)
            if problem.status == 'optimal':
                break
            else:
                print('Problem status not optimal: %s' % problem.status)
                print('trying another solver')
        except cvx.SolverError:
            print('solver {} failed.'.format(sl))

    if rw is None:
        print(('all solvers failed', solvers))
        x = np.ones(tdists.shape[1])
        return 0., 0., old_div(x, x.sum()), problem, sl
    elif problem.status != 'optimal':
        print(('Warning: problem status: %s', problem.status))

    ## get reward value without using approx
    achieved_rw = expected_reward_cvx2(tdists, palloc.value, future_steps, poisson_approx=False, **kwopts)
    return rw, achieved_rw, palloc.value, problem, sl

# # init weight array
# w = 1./len(w)
# while True:
#     part_id = weighted_sample(w) # pick part based on weights
#     frame_id = sample_part(part_id) # pick random frame within chosen part
#     has_result, res_id = black_box(frame_id) #instance detector (expensive)
#     w = update_weights(w, part_id, has_result, res_id)