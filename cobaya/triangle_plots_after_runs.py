from getdist import plots
import matplotlib.pyplot as plt

g=plots.get_subplot_plotter(chain_dir=r'/home/app/Git/LCDM-NS/cobaya')
roots = []
roots.append('cobaya without SSIM/run')
roots.append('cobaya with SSIM - CSD3/run')
roots.append('cobaya with iPPR - CSD3/run')
# roots.append('cobaya with SSIM - PC/run')
params = ['logA', 'n_s', 'theta_s_1e2', 'omega_b', 'omega_cdm', 'tau_reio']
g.triangle_plot(roots, params, filled=True, colors=['blue', 'grey', 'red'])
g.export()
plt.savefig('../illustrations/cosmology.pdf')
plt.show()

roots = []
roots.append('cobaya with SSIM - CSD3/run')
# roots.append('cobaya with iPPR - CSD3/run')
# roots.append('cobaya without SSIM/run')
roots.append('cobaya with SSIM - PC/run')
params = ['logA', 'n_s', 'theta_s_1e2', 'omega_b', 'omega_cdm', 'tau_reio']
g.triangle_plot(roots, params, filled=True, colors=['grey', 'orange'])
g.export()
plt.savefig('../illustrations/cosmo-pc.pdf')
plt.show()
