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
        return np.array([1/(b-a) if x > a and x < b else 0 for x in xs])
    else:
        return 1/(b-a) if xs > a and xs < b else 0


def N(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- ((x - mu)/sigma)**2)


def ppr(xs, mu, sigma, a, b, beta):
    Z, _ = integrate.quad(lambda x: N(x, mu, sigma)**beta, a, b)
    return np.array([N(x, mu, sigma)**beta if x > a and x < b else 0 for x in xs])/Z


def stochastic(xs, mu, sigma, a, b, beta):
    Z, _ = integrate.quad(lambda x: N(x, mu, sigma), a, b)
    rval = []
    for x in xs:
        h = hash(x)
        random.seed(h)
        r = random.random()
        if r > beta:
            rval.append(N(x, mu, sigma))
        else:
            rval.append(U(x, a, b))
    return np.array(rval)


def legends(**kwargs):
    plt.legend(**kwargs)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\pi(\theta)$')
