from uniform import BoxUniformModel
from resizeableUniform import ResizeableUniformPrior
from power_posterior import PowerPosteriorPrior
from general_mixture_model import StochasticMixtureModel
from true_gaussian import GaussianPeakedPrior
from numpy import array, mean, std
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from misc.dataseries import Series

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14
plt.rcParams["errorbar.capsize"] = 4


bounds = (-6*10**8, 6*10**8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]

preset = {
    'noResume': True,
    'nLive': 10
}

series = {}
series['uniform'] = Series(BoxUniformModel(*args), '.', r'$U$')
series['ppr'] = Series(PowerPosteriorPrior(*args), '+', r'$PPR$')
series['mix'] = Series(StochasticMixtureModel(
    [series['uniform'].model, GaussianPeakedPrior(*args)]), 'x', r'mix\((U, PPR)\)')


def nlike_calls(model, repeats, **kwargs):
    rval = []
    for _ in range(repeats):
        out, samples = model.exec_polychord(**kwargs)
        rval.append(out.nlike)
    return rval


def bench(repeats, nlike):
    rv = {}
    for k in series:
        data = []
        for nl in nlike:
            config = {
                'noResume': True,
                'nLive': nl
            }
            data.append(nlike_calls(series[k].model, repeats, **config))
        rv[k] = data
    return rv


nlive = [10, 80, 150, 300]
data = bench(3, nlive)


def compare(data):
    for k in series:
        xdata = nlive
        ydata = array([mean(x) for x in data[k]])
        yerr = array([std(x) for x in data[k]])
        plt.errorbar(xdata, ydata, yerr,
                     label=series[k].label, marker=series[k].style, markersize=8)
    plt.xlabel(r'\(n_{live}\)')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend()
    plt.show()


compare(data)
