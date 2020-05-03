import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14


def U(xs, a, b):
    if hasattr(xs, '__iter__'):
        return np.array([1/(b-a) if x > a and x < b else 0*x for x in xs])
    else:
        return 1/(b-a) if xs > a and xs < b else 0*xs


def N(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- ((x - mu)/sigma)**2)


def ppr(xs, mu, sigma, a, b, beta):
    Z, _ = integrate.quad(lambda x: N(x, mu, sigma)**beta, a, b)
    return np.array([N(x, mu, sigma)**beta if x > a and x < b else 0 for x in xs])/Z


def stochastic(xs, mu, sigma, a, b, beta):
    Z, _ = integrate.quad(lambda x: N(x, mu, sigma), a, b)
    rval = []
    if hasattr(xs, '__iter__'):
        for x in xs:
            h = hash(x)
            random.seed(h)
            r = random.random()
            if r > beta:
                rval.append(N(x, mu, sigma))
            else:
                rval.append(U(x, a, b))
        return np.array(rval)
    else:
        h = hash(xs)
        random.seed(h)
        r = random.random()
        if r > beta:
            return N(xs, mu, sigma)
        else:
            return U(xs, a, b)


def additive(xs, mu, sigma, a, b, beta):
    def f(x):
        return (beta)/(b-a) + (1-beta)*N(x, mu, sigma)
    Z, _ = integrate.quad(f, a, b)
    rval = []
    for x in xs:
        rval.append(f(x) if x > a and x < b else 0)
    return np.array(rval)


def legends(**kwargs):
    plt.legend(**kwargs)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\pi(\theta)$')
