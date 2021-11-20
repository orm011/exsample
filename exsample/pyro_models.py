from .sampler_common import *
import torch
# import torch.distributions as dist
import pyro.distributions as dists
from torch.distributions import constraints
import pyro.optim
import pyro.infer
import pandas as pd
from collections import deque

import pyro.contrib.gp as gp
from pyro.contrib.autoname import named
import torch
#import torch.distributions as dist
import pyro.distributions as dists
from torch.distributions import constraints
import pyro.optim
import pyro.infer
import pyro.contrib.autoguide as ag
import numpy as np

pyro.enable_validation(True)

class PyroModel(ScoringModel):
    def __init__(self, max_steps=100, **base_args):
        super().__init__(**base_args)
        self.prefix = 'PyroModel_%016x' % id(self)
        self.optim = pyro.optim.Adam({"lr": 0.01, "betas": (0.90, 0.999)})
        self.max_steps = max_steps
        self._params = None
        self.losses = []

    @property
    def S(self):
        return named.Object(self.prefix)

    def model(self, *args, **kwargs):
        raise NotImplementedError('implement model')

    def guide(self, *args, **kwargs):
        raise NotImplementedError('implement guide')

    def params(self):
        params = pyro.get_param_store().match(self.prefix)
        params2 = {a.replace(self.prefix + '.', ''): b for (a, b) in params.items()}
        return params2

    def update(self, Ns=None, Hs=None):

        if Ns is None or Hs is None:
            Ns = self.stats.Ns.float()
            Hs = self.stats.Hs.float()

        svi = pyro.infer.SVI(model=self.model,
                             guide=self.guide,
                             optim=self.optim,
                             loss=pyro.infer.Trace_ELBO())

        losses = deque([])
        for t in range(self.max_steps):
            loss = svi.step(Ns=Ns, Hs=Hs)
            losses.append(loss)

        self.losses.append(np.array(losses))

        # keep handle to all params (in case params cleared)
        # return self.params()

    def sample(self, batch_size):
        raise NotImplementedError('implement sample')

    def score(self, batch_size):
        assert self.randomize
        return self.sample(batch_size)


def make_cov(dts, sigma_t, range_c):
    cov = dts.div(sigma_t).pow(2.).neg().exp()
    cov = 0.5 * (cov + cov.t()) + 0.01 * torch.eye(cov.shape[0])
    cov = range_c * cov
    return cov


def pair_diff(x):
    all_pairs = torch.zeros(x.shape[0], x.shape[0])
    for i in range(all_pairs.shape[0]):
        for j in range(all_pairs.shape[1]):
            all_pairs[i, j] = (x[i] - x[j])
    return all_pairs


class TempModel(PyroModel):
    def __init__(self, splits, guide=None, **base_args):
        super().__init__(**base_args)
        ts = torch.tensor((splits.values.categories - splits.values.categories[0]).total_seconds())
        self.splits = splits
        self.ts = ts
        self.dts = pair_diff(ts)
        self.C = self.dts.shape[0]

        if guide is None:
            self._guide = ag.AutoMultivariateNormal(self.model, prefix=str(self.S.auto) + '.')
        else:
            self._guide = guide

        # call once to initialize. should be idempotent
        self._guide(Ns=torch.zeros(self.C), Hs=torch.zeros(self.C))

    def model(self, Ns, Hs=None):
        S = self.S
        sigma_t = S.sigma_t.param_(torch.tensor(60. * 30), constraints.positive)
        range_c = S.range_c.param_(torch.tensor(1.), constraints.positive)

        cov = make_cov(self.dts, sigma_t, range_c)
        theta_logit_dist = dists.MultivariateNormal(loc=torch.zeros(self.C), covariance_matrix=cov)
        logit_c = S.logit_c.sample_(theta_logit_dist)
        theta_c = torch.sigmoid(logit_c)

        with pyro.plate('chunk', self.C) as cind:
            obsdist = dists.Binomial(total_count=Ns[cind], probs=theta_c[cind])
            h_count = S.h_count.sample_(obsdist, obs=Hs)

        return S

    def guide(self, Ns, Hs=None):
        return self._guide(Ns, Hs)

    def sample(self, batch_size):
        raw = self._guide.get_posterior().sample(batch_size)
        return torch.sigmoid(raw)

# unsused, using autoguide now.
def temp_guide(Ns, dts, Hs=None):
    C = Ns.shape[0]

    # re-using?
    sigma_t = pyro.param('sigma_t', torch.tensor(60. * 30), constraints.positive)
    range_c = pyro.param('range_c', torch.tensor(1.), constraints.positive)

    alpha = pyro.param('alpha', torch.tensor(1.), constraints.positive)
    beta = pyro.param('beta', torch.tensor(1.), constraints.positive)

    cov = make_cov(dts, sigma_t, range_c)

    mu_est = pyro.param('v_mu', torch.zeros(C))
    sig_est = pyro.param('v_sig', torch.ones(C), constraints.positive)

    full_cov = alpha * cov + beta * torch.diag(sig_est)
    theta_logit_dist = dists.MultivariateNormal(loc=mu_est, covariance_matrix=full_cov)
    logit_c = pyro.sample('logit_c', theta_logit_dist)

    thdist = dists.TransformedDistribution(theta_logit_dist,
                                           [torch.distributions.transforms.SigmoidTransform()])

    return thdist