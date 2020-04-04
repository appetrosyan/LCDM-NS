import common_dists as dst
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

x = np.linspace(-1.5, 1.5, 60)


def add(x, beta):
    return dst.additive(x, 0, 0.5, -1, 1, beta)


plt.plot(x, add(x, 0), 'k:', label=r'$Add(U, G; \beta=0)$')
plt.plot(x, add(x, 0.3), 'k--', label=r'$Add(U, G;\beta=0.3)$')
plt.plot(x, dst.ppr(x, 0, 0.5, -1, 1, 0.3), 'k', label=r'$PPR(\beta=0.3)$')
plt.plot(x, add(x, 1), 'k-.', label=r'$Add(U, G; \beta=1)$')
plt.ylim(0, 2.5)
dst.legends(fontsize=12)

tikzplotlib.save('additive_mixtures.tex')
plt.show()
