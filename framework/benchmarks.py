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
from parmap import parmap
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14
plt.rcParams["errorbar.capsize"] = 4


bounds = (-6*10**8, 6*10**8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]

series = {}
series['uniform'] = Series(BoxUniformModel(*args), '.', r'$U$')
series['ppr'] = Series(PowerPosteriorPrior(*args), '+', r'$PPR$')
series['mix'] = Series(StochasticMixtureModel(
    [BoxUniformModel(*args), GaussianPeakedPrior(*args)]), 'x', r'mix\((U, G)\)')
series['gauss'] = Series(GaussianPeakedPrior(*args), 'o', r'$G$')

def nlike_calls(model, repeats, **kwargs):
    fr = kwargs.pop('file_root')
    outs = parmap(lambda r : model.exec_polychord(**kwargs, file_root=(fr + '{}'.format(r))), range(repeats))
    rval = [out[0].nlike for out in outs]
    print(rval)
    return rval


def bench(repeats, nlike):
    rv = {}
    config = {
        'noResume': True,
    }
    for k in tqdm(series):
        data=[nlike_calls(series[k].model, repeats, **config, nLive=nl, file_root='{}{}'.format(k, nl))
              for nl in nlike]
        rv[k] = data
    return rv


nlive = [10, 30, 40, 50, 55, 60, 65, 70]
data = bench(3, nlive)

def compare(data):
    for k in series:
        xdata = nlive
        ydata = array([mean(x) for x in data[k]])
        yerr = array([std(x) for x in data[k]])
        print('x={}, y={}, yerr={}'.format(xdata, ydata, yerr))
        plt.errorbar(xdata, ydata, yerr,
                     label=series[k].label, marker=series[k].style, markersize=8)
    plt.xlabel(r'\(n_{live}\)')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend()
    plt.show()


compare(data)
