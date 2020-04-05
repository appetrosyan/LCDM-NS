from misc.parallelism import parmap
from matplotlib import rc
import matplotlib.pyplot as plt
import tikzplotlib
from mpi4py import MPI
from numpy import array, sqrt, exp, pi, linspace
from gaussian_models.uniform import BoxUniformModel, ResizeableUniformPrior
from gaussian_models.power_posterior import PowerPosteriorPrior
from general_mixture_model import StochasticMixtureModel
from gaussian_models.true_gaussian import GaussianPeakedPrior

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14

bounds = (-6*10**8, 6*10**8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]
kwargs = {
    'noResume': True,
    'nLive': 20
}


ppr = PowerPosteriorPrior(*args, file_root='ppr')
tgd = GaussianPeakedPrior(*args, file_root='tgd')
bun = BoxUniformModel(*args, file_root='bun')
run = ResizeableUniformPrior(*args, file_root='run')

mix = StochasticMixtureModel([bun, tgd], file_root='mix')

qr, repart = run.exec_polychord(**kwargs)
q0, reference = bun.exec_polychord(**kwargs)
qg, gaussian = tgd.exec_polychord(**kwargs)
qp, power = ppr.exec_polychord(**kwargs)
qm, samples = mix.exec_polychord(**kwargs)

hist_samples = 1000


def overlay_gaussian(mu, sigma, **kwargs):
    x = linspace(-65, -60, 100)
    plt.fill_between(x, 0, hist_samples/50/(sigma * sqrt(2 * pi)) *
                     exp(- (x - mu)**2 / (2 * sigma**2)), **kwargs)


plt.hist(repart.logZ(hist_samples),
         label=r'Wrong \( \ln\  {\cal L} \)', hatch='-', fill=True, alpha=0.3)

# plt.hist(samples.logZ(hist_samples), label=r'mix\((U, G)\)', alpha=1)
overlay_gaussian(qm.logZ, qm.logZerr, label=r'mix\((U, G)\)', hatch='//')
# This would contain the right histogram, except polychord
# terminates before generating any useable data.
plt.hist(power.logZ(hist_samples), label='PPR', alpha=0.3)
# plt.hist(gaussian.logZ(hist_samples), label='\(G\) - true posterior', alpha=0.3, )
overlay_gaussian(qg.logZ, qg.logZerr,
                 label=r'$G$ - true posterior', hatch='|', alpha=0.3)
plt.hist(reference.logZ(hist_samples), label='\(U\) - reference',
         alpha=0.3, fill=True)


plt.xlabel(r'\(\ln {\cal Z}\)')
plt.ylabel(r'\(P(\ln {\cal Z})\)')
plt.legend()
tikzplotlib.save('../illustrations/histograms.tex')
plt.show()
anss = [qm, qr, qp, q0]

for x in anss:
    print('{:.2e}'.format(x.nlike))
