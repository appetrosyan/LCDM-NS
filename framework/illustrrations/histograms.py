import matplotlib.pyplot as plt
import tikzplotlib
import mpi4py
from matplotlib import rc
from numpy import array

from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.true_gaussian import GaussianPeakedPrior
from gaussian_models.uniform import BoxUniformModel
from gaussian_models.uniform import StrawManResizeablePrior
from general_mixture_model import StochasticMixtureModel

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams['font.size'] = 14

bounds = (-6 * 10 ** 8, 6 * 10 ** 8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
kwargs = {
    'noResume': True,
    'nLive': 200
}

ppr = PowerPosteriorPrior(bounds, mu, cov, file_root='ppr')
tgd = GaussianPeakedPrior(bounds, mu, cov, file_root='tgd')
bun = BoxUniformModel(bounds, mu, cov, file_root='bun')
run = StrawManResizeablePrior(bounds, mu, cov, file_root='run')
mix = StochasticMixtureModel([bun, tgd], file_root='mix')


qr, resizeable = run.nested_sample(**kwargs)
q0, reference = bun.nested_sample(**kwargs)
qp, power = ppr.nested_sample(**kwargs)
qm, samples = mix.nested_sample(**kwargs)

hist_samples = 1000
plt.hist(samples.logZ(hist_samples), label=r'mix\((U, G)\)', alpha=1)
plt.hist(resizeable.logZ(hist_samples),
         label=r'Wrong \( \ln\  {\cal L} \)', hatch='\\', fill=True, alpha=0.3)
plt.hist(power.logZ(hist_samples), label='PPR', alpha=0.3)
# plt.hist(gaussian.logZ(hist_samples),
#          label='\(G\) - true posterior', alpha=0.3, )
# For some reason this calculates the right evidence, but the histogram is shifted.
plt.hist(reference.logZ(hist_samples), label=r'\(U\) - reference',
         alpha=0.3, fill=True)

plt.xlabel(r'\(\ln {\cal Z}\)')
plt.ylabel(r'\(P(\ln {\cal Z})\)')
plt.legend()
tikzplotlib.save('../illustrations/histograms.tex')
plt.show()
answers = [qm, qr, qp, q0]

for x in answers:
    print(f'{x.nlike:.2e}')
