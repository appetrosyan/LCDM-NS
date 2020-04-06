import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib import rc
from numpy import array, mean, std

from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.true_gaussian import GaussianPeakedPrior
from gaussian_models.uniform import BoxUniformModel, StrawManResizeablePrior
from general_mixture_model import StochasticMixtureModel
from misc.data_series import Series
from misc.parallelism import parmap
from misc.ui import progressbar as tqdm
from offset_model import OffsetModel

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14
plt.rcParams["errorbar.capsize"] = 4


bounds = (-6*10**8, 6*10**8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]

coincidingSeries = {
    'uniform': Series(BoxUniformModel(*args), '.', r'$U$'),
    'ppr': Series(PowerPosteriorPrior(*args), '+', r'$PPR$'),
    'mix': Series(StochasticMixtureModel(
        [BoxUniformModel(*args),
         GaussianPeakedPrior(*args)]), 'x', r'mix\((U, G)\)'),
    'gauss': Series(GaussianPeakedPrior(*args), 'o', r'$G$')
}


def nlike_calls(model, repeats, **kwargs):
    fr = kwargs.pop('file_root')
    outs = parmap(lambda r: model.nested_sample(
        **kwargs, file_root=(fr + '{}'.format(r))), range(repeats))
    rval = [out[0].nlike for out in outs]
    print(rval)
    return rval


def bench(repeats, nlike, series):
    rv = {}
    config = {'noResume': True}
    for k in tqdm(series):
        print('Running {}'.format(k))
        nlikes = [nlike_calls(series[k].model,
                              repeats,
                              **config,
                              nLive=nl,
                              file_root='{}{}'.format(k, nl))
                  for nl in nlike]
        rv[k] = nlikes
    return rv


nlive = [10, 30, 40, 50, 55, 60, 65, 70]


def compare(data, nlike, series):
    for k in series:
        xdata = nlike
        ydata = array([mean(x) for x in data[k]])
        yerr = array([std(x) for x in data[k]])
        print('x={}, y={}, yerr={}'.format(xdata, ydata, yerr))
        plt.errorbar(xdata, ydata, yerr,
                     label=series[k].label, marker=series[k].style, markersize=8)
    plt.xlabel(r'\(n_{live}\)')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend()
    tikzplotlib.save('../illustrations/benchmark.tex')
    plt.show()


def generate_offset(series, factor=3):
    result = {}
    for k in series:
        result[k] = Series(OffsetModel(series[k].model, factor*mu),
                           style=series[k].style, label=series[k].label)
    return result


if __name__ == '__main__':
    main()


def main():
    # data = bench(3, nlive, coincidingSeries)
    # compare(data, coincidingSeries)
    global data, offsets
    offsets = generate_offset(coincidingSeries)
    data = bench(3, [10, 20], offsets)
    compare(data, [10, 20], offsets)
