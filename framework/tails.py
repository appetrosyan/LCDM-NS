from gaussian_models.power_posterior import PowerPosteriorPrior
from gaussian_models.uniform import BoxUniformModel
from numpy import array, diag
import matplotlib.pyplot as plt



mu = array([1,2,3])
cov = diag([1,1,1])
a = 6*10**8
bounds = (-array([a,a,a]), array([a,a,a]))

args = [bounds, mu, cov]
kwargs={
    'resume':False,
    'live_points': 20
}

models = {
    'uniform': BoxUniformModel(*args),
    'ppr': PowerPosteriorPrior(*args)
}

answers = {k: models[k].nested_sample(**kwargs) for k in models}
q = answers['uniform'][1]
