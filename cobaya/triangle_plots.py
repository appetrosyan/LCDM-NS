import matplotlib.pyplot as plt
import matplotlib
from getdist import plots
from getdist.chains import Chains
from getdist.mcsamples import MCSamples, loadMCSamples
import numpy as np
import pandas


matplotlib.use('Qt5Agg')
# g = plots.get_subplot_plotter(chain_dir=r'/home/app/Git/LCDM-NS/cobaya/cobaya_output')

chains = loadMCSamples(r'/home/app/Git/LCDM-NS/cobaya/cobaya_output/run')
# roots = ['run']
params = ['logA', 'n_s', 'theta_s_1e2', 'omega_b', 'omega_cdm', 'tau_reio']
# g.triangle_plot(roots, params, filled=True, shaded=True)
# plt.show()

print(repr(np.array(chains.getMeans())), repr(np.array(chains.getCov())))
np.savetxt(np.array(chains.getMeans()))
