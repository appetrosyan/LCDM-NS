import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib import rc
from numpy import array, mean, std

from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.true_gaussian import GaussianPeakedPrior
from gaussian_models.uniform import BoxUniformModel  # , StrawManResizeablePrior
from general_mixture_model import StochasticMixtureModel
from misc.data_series import Series
from misc.parallelism import parmap
from misc.ui import progressbar as tqdm

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14
plt.rcParams["errorbar.capsize"] = 4

bounds = (-6 * 10 ** 8, 6 * 10 ** 8)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]

series = {
    'uniform': Series(BoxUniformModel(*args), '.', r'$U$'),
    'ppr': Series(PowerPosteriorPrior(*args), '+', r'$PPR$'),
    'mix': Series(StochasticMixtureModel(
        [BoxUniformModel(*args),
         GaussianPeakedPrior(*args)]), 'x', r'mix\((U, G)\)'),
    'gauss': Series(GaussianPeakedPrior(*args), 'o', r'$G$')
}


def count_of_n_like_calls(model, repeats, **kwargs):
    fr = kwargs.pop('file_root')
    outs = parmap(lambda r: model.exec_polychord(
        **kwargs, file_root=(fr + f'{r}')), range(repeats))
    result = [out[0].nlike for out in outs]
    return result


def bench(repeats, nlike):
    rv = {}
    config = {'noResume': True}
    for k in tqdm(series):
        print(f'Running {k}')
        n_likes = [count_of_n_like_calls(series[k].model,
                                         repeats,
                                         **config,
                                         nLive=nl,
                                         file_root=f'{k}-{nl}')
                   for nl in nlike]
        rv[k] = n_likes
    return rv


nlive = [10, 30, 40, 50, 55, 60, 65, 70]
data = bench(3, nlive)


def compare(runs):
    for k in series:
        x_data = nlive
        y_data = array([mean(x) for x in runs[k]])
        y_err = array([std(x) for x in runs[k]])
        print(f'x={x_data}, y={y_data}, y_err={y_err}')
        plt.errorbar(x_data, y_data, y_err,
                     label=series[k].label, marker=series[k].style, markersize=8)
    plt.xlabel(r'\(n_{live}\)')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend()
    tikzplotlib.save('illustrations/benchmark.tex')
    plt.show()


compare(data)
