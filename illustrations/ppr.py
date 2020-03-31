from common_dists import legends, ppr
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

x = np.linspace(-1.5, 1.5, 120)

plt.plot(x, ppr(x, 0, 0.5, -1, 1, 0), 'k',
         label=r'$U(\theta; -1, 1) \equiv PPR(\theta; \beta=0)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, 0.3), 'k--',
         label=r'$PPR(\theta; \beta=0.3)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, 1), 'k-.',
         label=r'$PPR(\theta; \beta=1)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, -0.5), 'k:',
         label=r'$PPR(\theta; \beta=-0.5)$')
plt.ylim(0, 2.5)
legends(fontsize=12)


tikzplotlib.save('PPR.tex')
plt.show()
