import matplotlib.pyplot as plt
from nestcheck.data_processing import batch_process_data, process_polychord_run
from nestcheck.plots import param_logx_diagram
from numpy import array
from mpi4py import MPI
from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.uniform import BoxUniformModel
from general_mixture_model import StochasticMixtureModel

b = 10 ** 2
a = array([-b, -b, -b])
bounds = (a, -a)
mu = array([0, 4, 8])
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
file_roots = {k: k for k in models}
# answers = {k: models[k].nested_sample(file_root=file_roots[k], **kwargs) for k in models}

file_roots = [k for k in file_roots]

runs = batch_process_data(file_roots, parallel=False, process_func=process_polychord_run)
fig = param_logx_diagram(runs[:2],
                         logx_min=-20.00,
                         fthetas=[lambda x: x[:, 0], lambda x: x[:, 1], lambda x: x[:, 2]],
                         ftheta_lims={0: (-4, 5), 1: (0, 9), 2: (4, 13)})
plt.show()
