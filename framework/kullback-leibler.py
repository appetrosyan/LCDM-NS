import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib import rc
from numpy import array, mean, std, polyfit, concatenate, polyval, linspace, diag, sqrt
from mpi4py import MPI
from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.true_gaussian import GaussianPeakedPrior
from gaussian_models.uniform import BoxUniformModel
from general_mixture_model import StochasticMixtureModel
from misc.data_series import Series
from misc.parallelism import parmap
from misc.ui import progressbar as tqdm
from offset_model import OffsetModel

print(MPI)
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
         GaussianPeakedPrior(*args)]), 'x', r'mix\((U, G)\)')
}


def generate_offset(series, factor=3):
    result = {}
    for k in series:
        result[k] = Series(OffsetModel(series[k].model, factor * mu),
                           style=series[k].style, label=series[k].label)
    return result


def execute(model, repeats, **kwargs):
    fr = kwargs.pop('file_root')
    outs = parmap(lambda r: model.nested_sample(
        **kwargs, file_root=(fr + '{}'.format(r))), range(repeats))
    return outs


def bench(repeats, n_like, series):
    rv = {}
    config = {'resume': False}
    for k in tqdm(series):
        print('Running {}'.format(k))
        log_like_calls = [execute(series[k].model,
                                  repeats,
                                  **config,
                                  live_points=nl,
                                  file_root='{}{}'.format(k, nl))
                          for nl in n_like]
        rv[k] = log_like_calls
    return rv


def main():
    global offsets, data, ds, nlikes, offset_factors, outs, ds, calls, callerrs
    offset_factors = [10+10*x**2 for x in range(6)]
    offsets = [generate_offset(coincidingSeries, factor=x)
               for x in offset_factors]
    data = [bench(2, [20], offset) for offset in offsets]

    outs = [
        {k: ((mean([x[0].nlike for x in dat[k][0]]),
              std([x[0].nlike for x in dat[k][0]])),
             (mean([None if x[1] is None else x[1].D() for x in dat[k][0]]),
              std([None if x[1] is None else x[1].D() for x in dat[k][0]]))
             )
         for k in dat}
        for dat in data]
    outs = {k: [dic[k] for dic in outs] for k in outs[0]}
    for k in ['ppr', 'mix', 'uniform']:
        plt.errorbar(offset_factors, [x[1][0] for x in outs[k]],
                     [x[1][1] for x in outs[k]],
                     fmt=coincidingSeries[k].style,
                     label=coincidingSeries[k].label)
    plt.xlabel(r'Offset / $\vec{\mu}$')
    plt.ylabel(r'Kullback-Leibler Divergence $D$')
    plt.legend()
    tikzplotlib.save('../illustrations/kullback-leibler.tex')
    plt.show()

    for k in ['ppr', 'mix', 'uniform']:
        plt.errorbar([x[1][0] for x in outs[k]],
                     [x[0][0] for x in outs[k]],
                     xerr=[x[1][1] for x in outs[k]],
                     yerr=[x[0][1] for x in outs[k]],
                     fmt=coincidingSeries[k].style,
                     label=coincidingSeries[k].label)

    ds = concatenate([[x[1][0] for x in outs[k]] for k in outs])
    calls = concatenate([[x[0][0] for x in outs[k]] for k in outs])
    callerrs = concatenate([[x[0][0] for x in outs[k]] for k in outs])
    p, cov = polyfit(ds, calls, deg=1, w=1/callerrs, cov=True)
    sigma = sqrt(diag(cov))
    sigma = sigma/10**3
    p = p/10**3
    bfl = f'\\left[({p[1]:2.1f} \\pm {sigma[1].round(2):1.1f})x + ({p[0]:2.1f} \\pm {sigma[0]:1.1f})\\right]\\cdot  10^3'
    p = p*10**3
    sigma = sigma*10**3
    xs = linspace(min(ds), max(ds), 100)
    plt.plot(xs, polyval(p, xs), label=r'\('+bfl+r' \)')
    plt.xlabel(r'Kullback-Leibler Divergence $D$')
    plt.ylabel(r'\# of \({\cal L}\) evaluations')
    plt.legend(loc='upper left')
    tikzplotlib.save('../illustrations/scaling-kld.tex')
    plt.show()

    outs = [
        {k: ((mean([x[0].logZ for x in dat[k][0]]),
              std([x[0].logZ for x in dat[k][0]])),
             (mean([None if x[1] is None else x[1].D() for x in dat[k][0]]),
              std([None if x[1] is None else x[1].D() for x in dat[k][0]]))
             )
         for k in dat}
        for dat in data]
    outs = {k: [dic[k] for dic in outs] for k in outs[0]}
    for k in ['ppr', 'mix', 'uniform']:
        plt.errorbar(offset_factors,
                     [x[0][0] for x in outs[k]],
                     yerr=[x[0][1] for x in outs[k]],
                     marker=coincidingSeries[k].style,
                     label=coincidingSeries[k].label)
    plt.xlabel(r'Offset / $\vec{\mu}$')
    plt.ylabel(r'$\ln {\cal Z}$')
    plt.legend()
    tikzplotlib.save('../illustrations/evidence-drift.tex')
    plt.show()


if __name__ == '__main__':
    main()
