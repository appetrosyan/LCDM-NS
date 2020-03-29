from common_dists import legends, ppr
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

x = np.linspace(-1.5, 1.5, 60)

plt.plot(x, ppr(x, 0, 0.5, -1, 1, 0), 'k',
         label=r'$U(\theta; -1, 1) = PPR(\theta; \beta=0)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, 0.5), 'k--',
         label=r'$G(\theta, \mu=0, \sigma=0.5)^{\beta} = PPR(\theta; \beta=0.5)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, 1), 'k-.',
         label=r'$G(x, \mu=0, \sigma=0.5)$')
plt.plot(x, ppr(x, 0, 0.5, -1, 1, -0.5), 'k:',
         label=r'$PPR(\theta; \beta=-0.5)$')
legends()

tikzplotlib.save('PPR.tex')
plt.show()
