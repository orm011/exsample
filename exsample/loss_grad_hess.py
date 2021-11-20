from jax import numpy as jnp
from jax import grad
import jax.scipy.special
import numpy

def check_w(w):
    assert jnp.isclose(w.sum(), 1.)
    assert (w >= 0).all()


class DirectL(object):
    def __init__(self, P, n=1):
        self.P = P
        self.n = n

    def _ell(self, w):
        n, P = (self.n, self.P)
        return jnp.exp(-n * (P @ w))

    def F(self, w):
        return self._ell(w).sum()

    def gradF(self, w):
        n, P = (self.n, self.P)
        el = self._ell(w).reshape(-1, 1)
        return -n * (el * P).sum(axis=0)

    def hessF(self, w):
        n, P = (self.n, self.P)
        Lmat = jnp.diag(self._ell(w))
        return (n ** 2) * (P.transpose() @ Lmat @ P)


class LogL(object):
    def __init__(self, P, n=1):
        self.P = P
        self.n = n

    def F(self, w):
        n, P = (self.n, self.P)
        log_ell = -n * (P @ w)
        return jax.scipy.special.logsumexp(log_ell)

    def gradF(self, w):
        n, P = (self.n, self.P)
        s = jax.nn.softmax(-n * (P @ w), axis=0).reshape(-1, 1)
        return -n * (s * P).sum(axis=0)

    def hessF(self, w):
        n, P = (self.n, self.P)
        s = jax.nn.softmax(-n * (P @ w))
        sprime = jnp.diag(s) - jnp.outer(s, s)
        # s has tiny entries when n is large, there are many instances.
        # outer(s,s) therefore has even tinier entries.
        # may benefit from working with logsoftmax.
        return (n ** 2) * (P.transpose() @ sprime @ P)


class LogdomainL(object):
    def __init__(self, P, n):
        self.P = P
        self.n = n
        self.logL = LogL(P, n)

    def F(self, w):
        return jnp.exp(self.logL.F(w))

    def gradF(self, w):
        glogl = self.logL.gradF(w)
        return self.F(w) * glogl

    def hessF(self, w):
        f = self.F(w)
        gradF = self.gradF(w)
        return ((f ** 2) * self.logL.hessF(w) + jnp.outer(gradF, gradF)) / f


class JaxBasedL(object):
    def __init__(self, func):
        self.func = func

    def F(self, w):
        return self.func(w)

    def gradF(self, w):
        return grad(self.func)(w)

    def hessF(self, w):
        def hvp(f, x, v):
            return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)

        vs = jnp.eye(w.shape[0])
        hs = []
        for v in vs:
            h = hvp(self.func, w, v)
            hs.append(h)

        return jnp.stack(hs, axis=1)


def make_testws(P, num_samples):
    assert num_samples >= 4
    uniform_w = numpy.ones((1, P.shape[1]))
    random_w = numpy.random.randn(num_samples // 2 - 1, P.shape[1])
    unnormalized_ws = jnp.concatenate([uniform_w, random_w])
    normalized_ws = jax.nn.softmax(unnormalized_ws, axis=-1)
    ws = jnp.concatenate([normalized_ws, unnormalized_ws])
    return ws

from tqdm.auto import tqdm

def test_values(P, fun, refF, max_n=10, num_samples=10):
    ws = make_testws(P, num_samples=num_samples)
    for n in tqdm(range(max_n + 1)):
        for w in ws:
            out = fun(P, n).F(w)
            ref = refF(P, n).F(w)
            assert jnp.isclose(ref, out).all()


def test_grads(P, lfun, max_n=10, num_samples=5):
    ws = make_testws(P, num_samples=num_samples)
    for n in tqdm(range(max_n + 1)):
        for w in ws:
            l = lfun(P, n)
            ref = JaxBasedL(l.F).gradF(w)
            out = l.gradF(w)
            clmat = jnp.isclose(ref, out)
            assert clmat.all()


def test_hessian(P, lfun, max_n=10, num_samples=5):
    ws = make_testws(P, num_samples=num_samples)
    for n in tqdm(range(max_n)):
        for w in ws:
            l = lfun(P, n)
            ref = JaxBasedL(l.F).hessF(w)
            out = l.hessF(w)
            clmat = jnp.isclose(ref, out, atol=1e-7)
            assert clmat.all()

def test(P):
    test_values(P,LogdomainL,DirectL,max_n=10,num_samples=10)
    test_grads(P,DirectL,max_n=5, num_samples=10)
    test_grads(P,LogL,max_n=5, num_samples=10)
    test_grads(P,LogdomainL,max_n=5, num_samples=10)
    test_hessian(P,DirectL,max_n=5, num_samples=10)
    test_hessian(P,LogdomainL,max_n=5, num_samples=10)
    test_hessian(P,LogL,max_n=5,num_samples=10)