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

# The fact that this doesn't work, is living proof that Anthony Lewis should stick to science.
# This is very poorly thought out, and I'm surprised so many people are using any of his programs. 
# print(chains.getMeans(['logA']))

chains.getParamNames().saveAsText('paramnames')
np.savetxt('means', chains.getMeans(range(27)))

np.savetxt('covs', chains.getCov(pars=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]))
