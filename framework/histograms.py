from uniform import BoxUniformModel
from resizeableUniform import ResizeableUniformPrior
from power_posterior import PowerPosteriorPrior
from general_mixture_model import StochasticMixtureModel
from true_gaussian import GaussianPeakedPrior
from numpy import array
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14

bounds = (-6*10**8, 6*10**8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
kwargs = {
    'noResume': True,
    'nLive': 20
}


ppr = PowerPosteriorPrior(bounds, mu, cov)
tgd = GaussianPeakedPrior(bounds, mu, cov)
bun = BoxUniformModel(bounds, mu, cov)
run = ResizeableUniformPrior(bounds, mu, cov)
mix = StochasticMixtureModel([bun, tgd])

qr, repart = run.exec_polychord(**kwargs)
q0, reference = bun.exec_polychord(**kwargs)
# qg, gaussian = tgd.exec_polychord(**kwargs)
qp, power = ppr.exec_polychord(**kwargs)
qm, samples = mix.exec_polychord(**kwargs)

hist_samples = 1000
plt.hist(samples.logZ(hist_samples), label=r'mix\((U, G)\)', alpha=1)
plt.hist(repart.logZ(hist_samples),
         label=r'Wrong \( \ln\  {\cal L} \)', hatch='\\', fill=True, alpha=0.3)
plt.hist(power.logZ(hist_samples), label='PPR', alpha=0.3)
# plt.hist(gaussian.logZ(hist_samples),
#          label='\(G\) - true posterior', alpha=0.3, )
# For some reason this calculates the right evidence, but the histogram is shifted.
plt.hist(reference.logZ(hist_samples), label='\(U\) - reference',
         alpha=0.3, fill=True)


plt.xlabel(r'\(\ln {\cal Z}\)')
plt.ylabel(r'\(P(\ln {\cal Z})\)')
plt.legend()
tikzplotlib.save('../illustrations/histograms.tex')
plt.show(block=False)
anss = [qm, qg, qr, qp, q0]

for x in anss:
    print('{:.2e}'.format(x.nlike))
