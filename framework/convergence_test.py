import matplotlib.pyplot as plt
from anesthetic.plot import get_legend_proxy
from numpy import array
from mpi4py import MPI
from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.uniform import BoxUniformModel
from general_mixture_model import StochasticMixtureModel
from offset_model import OffsetModel

print(MPI)
b = 10 ** 3
a = array([-b, -b, -b])
bounds = (a, -a)
mu = array([1, 2, 3])
cov = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
args = [bounds, mu, cov]
kwargs = {
    'live_points': 120,
    'resume': False
}

models = {
    'ppr': PowerPosteriorPrior(*args),
    'uniform': BoxUniformModel(*args)
}
models['mix'] = StochasticMixtureModel([models['ppr'], BoxUniformModel(*args)])
offsets = {k: OffsetModel(models[k], mu * 2) for k in models}
answers = {
    'model': {k: models[k].nested_sample(**kwargs) for k in models},
    'offset': {k: offsets[k].nested_sample(**kwargs) for k in offsets}
}
labels = []
alpha = 0.7
fig, ax = answers['offset']['ppr'][1].plot_2d(
    [0, 1], alpha=alpha, color='#A000A0')
labels.append('Offset $PPR$')
answers['model']['ppr'][1].plot_2d(ax, alpha=alpha, color='#FF0000')
labels.append('Model $PPR$')
answers['offset']['mix'][1].plot_2d(ax, alpha=alpha, color='#A0A000')
labels.append('Offset $mix$')
answers['model']['mix'][1].plot_2d(ax, alpha=alpha, color='#00FF00')
labels.append('Model $mix$')
answers['offset']['uniform'][1].plot_2d(ax, alpha=alpha, color='#00A0A0')
labels.append('Offset $U$')
answers['model']['uniform'][1].plot_2d(ax, alpha=alpha, color='#0000FF')
labels.append('Model $U$')

proxy = get_legend_proxy(fig)
fig.legend(proxy, labels)
plt.show()
