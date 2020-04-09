import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib import rc
from numpy import array, mean, std
from mpi4py import MPI
from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.true_gaussian import GaussianPeakedPrior
from gaussian_models.uniform import BoxUniformModel
from general_mixture_model import StochasticMixtureModel
from misc.data_series import Series
from misc.parallelism import parmap
from misc.ui import progressbar as tqdm
from offset_model import OffsetModel

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 14
plt.rcParams["errorbar.capsize"] = 4

bounds = (-6 * 10 ** 8, 6 * 10 ** 8)
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


def count_log_like_calls(model, repeats, **kwargs):
    fr = kwargs.pop('file_root')
    outs = parmap(lambda r: model.nested_sample(
        **kwargs, file_root=(fr + '{}'.format(r))), range(repeats))
    result = [out[0].nlike for out in outs]
    print(result)
    return result


def bench(repeats, n_like, series):
    rv = {}
    config = {'noResume': True}
    for k in tqdm(series):
        print('Running {}'.format(k))
        log_like_calls = [count_log_like_calls(series[k].model,
                                               repeats,
                                               **config,
                                               nLive=nl,
                                               file_root='{}{}'.format(k, nl))
                          for nl in n_like]
        rv[k] = log_like_calls
    return rv


live_points = [10, 30, 40, 50, 55, 60, 65, 70]


def compare(runs, n_like, series):
    for k in series:
        x_data = n_like
        y_data = array([mean(x) for x in runs[k]])
        y_err = array([std(x) for x in runs[k]])
        print('x={}, y={}, y_err={}'.format(x_data, y_data, y_err))
        plt.errorbar(x_data, y_data, y_err,
                     label=series[k].label, marker=series[k].style, markersize=8)
    plt.xlabel(r'\(n_{live}\)')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend()
    tikzplotlib.save('../illustrations/benchmark.tex')
    plt.show()


def generate_offset(series, factor=3):
    result = {}
    for k in series:
        result[k] = Series(OffsetModel(series[k].model, factor * mu),
                           style=series[k].style, label=series[k].label)
    return result


def main():
    # runs = bench(3, n_live, coincidingSeries)
    # compare(runs, coincidingSeries)
    offsets = generate_offset(coincidingSeries)
    data = bench(3, [10, 20], offsets)
    compare(data, [10, 20], offsets)


if __name__ == '__main__':
    main()
