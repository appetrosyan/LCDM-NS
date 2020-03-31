# FUCK YOU FOR NOT DEBUGGING THIS THING!!!!

import common_dists as dist
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np


x = np.linspace(-1.5, 1.5, 60)


def N(x):
    return dist.N(x, 0, 0.5)


def U(x):
    return dist.U(x, -1, 1)


def stoch(x, beta):
    return dist.stochastic(x, 0, 0.5, -1, 1, beta)


plt.plot(x, N(x), 'k:', label=r'$G(\theta; \mu=0, \sigma=0.5)$')
plt.plot(x, U(x), 'k--', label=r'$U(\theta, -1, 1)$')
plt.plot(x, stoch(x, 0.5), 'k+', label=r'mix$(G, U; \beta=0.5)$')

dist.legends(fontsize=12)
plt.ylim(0, 1.2)
plt.xlim(-1.4, 1.6)
tikzplotlib.save('mixture.tex')
plt.show()
