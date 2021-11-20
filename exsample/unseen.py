from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
import torch as t


def get_fingerprint(samples):
    samples = (samples - samples.min())
    by_id = torch.histc(samples.float(), min=0, max=samples.max() + 1, bins=samples.max() + 1)
    max_count = by_id.long().max()
    f = torch.histc(by_id, min=0, max=max_count + 1, bins=max_count.item() + 1)
    f[0] = 0.

    assert (f.dot(t.arange(f.numel()).float()) == samples.numel()).item()
    assert f[0] == 0
    return f

def paper_defaults(f, grid_factor=1.05, grid=None):
    """
    implements default grid and default split into LP section and empirical section
    as per matlab code.
    """
    assert f[0] == 0
    szs = np.arange(0, f.shape[0])
    k = np.dot(szs, f)

    """
    xLPmin = 1/(k*max(10,k)); 
    min_i=min(find(f>0));
    if min_i > 1
        xLPmin = min_i/k;
    end
    """
    xLPmin = 1. / (k * max(10, k))
    min_i = np.min(np.nonzero(f))
    if min_i > 1:
        xLPmin = old_div(min_i, k)
    """
    % Split the fingerprint into the 'dense' portion for which we 
    % solve an LP to yield the corresponding histogram, and 'sparse' 
    % portion for which we simply use the empirical histogram
    x=0;
    histx = 0;
    fLP = zeros(1,max(size(f)));
    for i=1:max(size(f))
        if f(i)>0
            wind = [max(1,i-ceil(sqrt(i))),min(i+ceil(sqrt(i)),max(size(f)))];
            if sum(f(wind(1):wind(2)))<sqrt(i)% 2*sqrt(i)
                x=[x, i/k];
                histx=[histx,f(i)];
                fLP(i)=0;
            else
                fLP(i)=f(i);
            end
        end
    end
    """
    x = []
    histx = []
    fLP = np.zeros_like(f)
    for i in range(f.shape[0]):
        if f[i] > 0:
            delta_i = int(np.ceil(np.sqrt(i)))
            w_start, w_end = [max(1, i - delta_i), min(i + delta_i, f.shape[0])]
            if np.sum(f[w_start:w_end]) < np.sqrt(i):
                x.append(i)
                histx.append(f[i])
            else:
                fLP[i] = f[i]
    x = np.array(x) / 1.0 / k  # divide for prob.
    histx = np.array(histx)
    """
    % If no LP portion, return the empirical histogram
    fmax = max(find(fLP>0));
    if min(size(fmax))==0
        x=x(2:end);
        histx=histx(2:end);
        return;
    end
    """
    nz = np.nonzero(fLP)[0]
    if nz.shape[0] == 0:
        return {'empirical': (x, histx), 'LP': None}

    fmax = np.max(nz)

    """
    % Set up the first LP
    LPmass = 1 - x*histx'; %amount of probability mass in the LP region

    fLP=[fLP(1:fmax), zeros(1,ceil(sqrt(fmax)))];
    szLPf=max(size(fLP));

    xLPmax = fmax/k;
    xLP=xLPmin*gridFactor.^(0:ceil(log(xLPmax/xLPmin)/log(gridFactor)));
    szLPx=max(size(xLP));
    """
    padding = int(np.ceil(np.sqrt(fmax)))
    fLP = np.concatenate([fLP[:fmax + 1], np.zeros(padding)])
    xLPmax = fmax / 1.0 / k
    steps = int(np.ceil(old_div((np.log(xLPmax) - np.log(xLPmin)), np.log(grid_factor))))
    xLP = xLPmin * np.power(grid_factor, np.arange(0., steps + 1))
    assert xLP[0] == xLPmin
    assert xLP[-1] >= xLPmax
    assert xLP[-2] < xLPmax
    assert np.abs(old_div((xLP[1] /1.0/ xLP[0]),grid_factor) - 1.) < 0.01

    if grid is not None:
        xLP = grid[grid >= xLPmin]
        xLP = xLP[xLP<= xLPmax]

    return {'empirical': (torch.tensor(x), torch.tensor(histx)), 'LP': (torch.tensor(xLP), torch.tensor(fLP))}


import torch
import torch.distributions as dist
import cvxpy as cvx
import numpy as np


def get_matrix(sample_hist, grid):
    hist = sample_hist

    assert torch.is_tensor(hist)
    assert torch.is_tensor(grid)
    assert hist[0] == 0  # no samples are observed zero times
    #assert (grid <= 1.).all()
    assert (grid > 0).all()

    counts = torch.arange(hist.shape[0]).double()
    N = hist.dot(counts)
    poisson_lambdas = grid * N.double()
    dists = dist.Poisson(rate=poisson_lambdas)
    probs_per_rate = dists.log_prob(counts.view(-1, 1)).exp()
    return probs_per_rate


def problem_params(sample_hist, grid, as_numpy=False):
    hist = sample_hist.double()
    assert hist[0] == 0
    Nsamples = hist.dot(torch.arange(0,hist.shape[0]).double())
    M = get_matrix(hist, grid)
    w = 1. / (hist + 1.).sqrt()

    if not as_numpy:
        return M, Nsamples, w
    else:
        return M.numpy(), Nsamples.numpy().item(), w.numpy()


def as_histogram(any_vec, grid):
    """
        converts any real vec into a valid histogram (used to implement constraints)
    """
    gridl = grid.log()
    renorm = (any_vec + gridl).log_softmax(dim=-1)
    return (renorm - gridl).exp()


def pt_loss(M, Nsamples, w, sample_hist, composition):
    assert sample_hist[0] == 0
    assert M.shape[0] == sample_hist.shape[0]

    expected_f = M.mv(composition)
    deviations = (expected_f[1:] - sample_hist[1:]) * w[1:]
    loss = deviations.abs().sum()
    return loss


def cvx_loss(M, w, sample_hist, composition, quadratic=False):
    expected_f = cvx.matmul(M, composition)
    deviations = (expected_f - sample_hist)
    wdev = cvx.multiply(deviations, w)
    devs = wdev[1:]

    if not quadratic:
        loss = cvx.sum(cvx.abs(devs))
    else:
        loss = cvx.sum_squares(devs)
    return loss


def cvx_prob(M, Nsamples, w, sample_hist, grid):
    composition = cvx.Variable(M.shape[1])
    loss = cvx_loss(M, w, sample_hist, composition)
    grid = grid.numpy()
    constraints = [composition >= 0.,
                   cvx.matmul(composition, grid) == 1.]
    prob = cvx.Problem(objective=cvx.Minimize(loss), constraints=constraints)
    return loss, composition, prob


def pad_hist(hist):
    nz = torch.nonzero(hist)
    min_c = nz.min()
    max_c = nz.max()
    assert min_c > 0
    padding_len = max_c.double().sqrt().ceil().int()
    hist = torch.cat([hist[:max_c+1], torch.zeros(padding_len).double()])
    return hist


def gengrid(minP, maxP, gridFactor=1.05):
    exps = torch.arange(0., np.log(old_div(maxP * gridFactor, minP)), step=np.log(gridFactor))
    ret = minP * exps.exp()
    assert ret[0] == minP
    assert ret[-1] >= maxP
    return ret


class OptHist(object):
    def __init__(self, sample_hist, grid, condition=True, prev_h=None, LPmass=1.):
        assert sample_hist[0] == 0  # zero index
        self.sample_hist = pad_hist(sample_hist).double().numpy()
        self.grid = grid
        M, N, w = problem_params(grid=self.grid, sample_hist=sample_hist)

        self.M = M.numpy()
        self.G = torch.ones_like(self.grid).numpy()
        self.g = torch.tensor(self.grid).numpy() #.grid.copy()


        if condition:
            self.M = old_div(self.M, self.g)  # meant to be row-wise]
            self.G =  old_div(self.G, self.g)
            self.g = old_div(self.g, self.g)


        self.N = N
        self.w = w.numpy()
        self.eps = 0.1
        self.LPmass = LPmass
        self.prev_h = prev_h


    def stage1_pytorch(self, init_logits=None, N=20001):
        assert self.LPmass == 1., "need to adapt the softmax otherwise"
        from torch.optim import Adam

        if init_logits is None:
            logits = torch.randn_like(self.grid, requires_grad=True)
        else:
            logits = init_logits

        opt = Adam(params=[logits], lr=0.01)
        best_sol = logits
        best_loss = np.inf

        for i in range(N):
            opt.zero_grad()
            as_hist = as_histogram(logits, self.grid)
            l = self.loss(as_hist)

            if l < best_loss:
                best_loss = l.detach().item()
                best_sol = logits.detach()

            if i % 5000 == 0:
                print(i, 'loss ', l.detach().item(), ' best loss ', best_loss)

            l.backward()
            opt.step()
        return as_histogram(best_sol, self.grid)

    def loss(self, candidate):
        return pt_loss(M=self.M, Nsamples=self.N,
                       sample_hist=self.sample_hist,
                       w=self.w, composition=torch.tensor(candidate))

    def check_valid(self, candidate):
        #assert (candidate >= -self.eps).all(), candidate
        sol = candidate*self.G
        lpmass_err = np.abs(np.dot(sol, self.grid) - self.LPmass)
        assert np.abs(lpmass_err) <= self.eps, lpmass_err
        assert np.sum(sol) >= 0

    def stage1(self, quadratic=False, **kwargs):
        composition = cvx.Variable(self.M.shape[1])
        loss = cvx_loss(M=self.M, w=self.w, sample_hist=self.sample_hist,
                        composition=composition, quadratic=quadratic)

        prob = cvx.Problem(objective=cvx.Minimize(loss),
                           constraints=[composition >= 0.,
                                        cvx.matmul(composition, self.g) == self.LPmass]

                           )

        lval = prob.solve(**kwargs)
        if quadratic:
            lval = np.sqrt(lval)
        return composition.value, lval

    def stage2(self, composition, lval, alpha, quadratic=False, **kwargs):
        loss_ceiling = lval + alpha
        composition2 = cvx.Variable(composition.shape)
        new_loss = cvx_loss(M=self.M,  w=self.w,
                            sample_hist=self.sample_hist,
                            composition=composition2, quadratic=quadratic)

        support = cvx.matmul(composition2, self.G)
        # if condition, G should be 1/grid.
        # else, G should be 1, since we onlu sum

        prob2 = cvx.Problem(cvx.Minimize(support),
                            constraints=[composition2 >= 0.,
                                         cvx.matmul(composition2, self.g) == self.LPmass,
                                         new_loss <= loss_ceiling])

        composition2.value = composition
        kwargs['warm_start'] = True
        sup = prob2.solve(**kwargs)
        actual_loss = cvx_loss(M=self.M,  w=self.w,
                            sample_hist=self.sample_hist,
                            composition=composition2, quadratic=quadratic)
        return composition2.value, sup, actual_loss.value



    def estimate_histogram(self, alpha=0.5, quadratic=False, **kwargs):
        c1, lval1 = self.stage1(quadratic=quadratic, **kwargs)
        self.check_valid(c1)
        c2, sup, lval2 = self.stage2(c1, lval1, alpha=alpha, quadratic=quadratic, **kwargs)
        self.check_valid(c2)

        delta_err = lval2 - lval1
        err_margin = 1e-4
        assert delta_err <= alpha + err_margin
        assert delta_err >= 0 - err_margin
        return torch.tensor(c2*self.G)


def expected_reward_future(h, x, nsteps):
    p_hit = 1. - (1. - x).pow(nsteps)
    return h.dot(p_hit)

def expected_reward_hist(h, x, N, nsteps=1):
    p_miss = (1.- x).pow(N)
    unseen_h = h*p_miss
    return expected_reward_future(unseen_h, x, nsteps)

default_opts = dict(
    condition=True,
    solver='GLPK', # some other solvers seem to yield solutions not quite within constraints
    verbose=True,
    #max_iter=8000, # not useful
    alpha=0.5,
    grid_factor=1.05,
    quadratic=False
)



def paper_unseen(f, grid_factor=1.05, condition=True,
                    alpha=0.7,
                    quadratic=False,
                    **opts):
    df = paper_defaults(f.numpy(), grid_factor=grid_factor)
    h = None
    (emp_grid, emp_hist) = df['empirical']
    if df['LP'] is None:
        x = torch.tensor(emp_grid)
        h = torch.tensor(emp_hist)
        return h, x, None, None, None

    (lp_grid, lp_hist) = df['LP']

    emp_mass = emp_grid.dot(emp_hist).item()
    lp_mass = 1. - emp_mass


    oh = OptHist(grid=lp_grid.double(), condition=condition,
                 sample_hist=lp_hist.double(), prev_h=h, LPmass=lp_mass)
    h = oh.estimate_histogram(**opts)
    x = oh.grid
    assert np.abs(x.dot(h) - lp_mass) < 0.01, x.dot(h) - lp_mass

    full_h = torch.cat([emp_hist, h], dim=-1)
    full_x = torch.cat([emp_grid, x], dim=-1)
    return h,x,oh,full_h,full_x



def generate_pmatrix(scales, period, fmax):
    max_c = ((old_div(scales, period))).ceil()
    p_max = (old_div(scales, period)) - max_c + 1

    max_count = int(max_c.max().item())
    matrix = torch.zeros(fmax, scales.shape[0])

    for (j, m) in enumerate(p_max.int()):
        matrix[m - 1, j] = 1. - p_max[j]
        matrix[m, j] = p_max[j]
    return matrix


def bump_unseen(numpy_f, period, cvx_opts={}):
    for i,_ in enumerate(numpy_f):
        if i < 2:
            continue

        if numpy_f[i] == 0:
            break


    fmax = i
    numpy_f = np.concatenate([numpy_f[:fmax], [0.]])

    max_scale = (fmax+1)*period #eg. seen something twice at a rate of 10fs. can be 30 but not 31
    scales = np.arange(max_scale) + 1
    m = generate_pmatrix(scales, period, numpy_f.shape[0]).numpy()

    est = cvx.Variable(m.shape[1])
    expected = cvx.matmul(m, est)
    weights = 1./np.sqrt(numpy_f + 1.) ## penalize stuff w f
    weights[:] = 1.
    weights[0] = 0. # don't penalize unseen stuff

    loss = cvx.sum(cvx.abs((numpy_f - expected)))
    prob = cvx.Problem(cvx.Minimize(loss), constraints=[est >= 0])
    val =  prob.solve(**cvx_opts)
    ns = est.value

    ## now what is the expected results under a doubling?
    m2 = generate_pmatrix(scales, period//2, numpy_f.shape[0]).numpy()
    efp = m2.dot(ns)
    egain = expected.value[0] - efp[0]
    return egain, expected.value, efp, ns, prob

