import numpy
# import pypolychord
from pypolychord import run_polychord
import pypolychord.priors
import pypolychord.settings
from scipy.special import erfinv
from anesthetic import NestedSamples
import matplotlib.pyplot as plt
import time

mu = numpy.array([1.0, 2.0, 3.0, 4.0])
cov = numpy.array([[1.0, 0.6, 0.6, 0.6],
                   [0.6, 1.0, 0.6, 0.6],
                   [0.6, 0.6, 1.0, 0.6],
                   [0.6, 0.6, 0.6, 1.0]])
nDims = len(mu)
nDerived = 0

settings = pypolychord.settings.PolyChordSettings(nDims + 1, nDerived)
# settings.file_root = 'temperature_1'
settings.nlive = 200            # Should technically be nlive
settings.read_resume = False
settings.do_clustering = True
# settings.feedback = 0
L = numpy.linalg.cholesky(cov)
root2 = numpy.sqrt(2)


def zed(beta):
    return beta ** (nDims/2)


def prior(point_in_hypercube):
    pil = point_in_hypercube[:-1]
    beta = point_in_hypercube[-1:].item()
    theta = mu + root2 * L @ erfinv(2*(pil) ** (beta)  # / zed(beta)
                                    - 1)
    return numpy.append(theta, beta)


def logl(theta):
    theta_ = theta[:-1]
    beta = theta[-1:].item()
    invSig = numpy.linalg.inv(cov)
    norm = numpy.linalg.slogdet(2*numpy.pi*cov)[1]/2
    logL = - norm - \
        numpy.linalg.multi_dot([(theta_ - mu), invSig, (theta_-mu)])
    #  For some reason uncommenting this part makes it run slower!
    # logL = (logL)*beta - (nDims/2) * numpy.log(beta)
    return logL, []


def bench(func, *args, repeats=1):
    times = []
    for i in range(repeats):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end-start)
    return times


settings.feedback = 0

settings.file_root = 'power-posterior'
times_repart = bench(run_polychord, logl, nDims + 1,
                     0, settings, prior, repeats=1)

# [16.387171983718872, 16.230791091918945, 16.40148901939392,
# 16.367916107177734, 17.56959104537964, 16.267454385757446,
# 16.094260215759277, 16.215574979782104, 16.516545057296753,
# 17.47413992881775]

settings.file_root = 'uniform'
times_uniform = bench(run_polychord, logl, nDims + 1, 0, settings,
                      pypolychord.priors.UniformPrior(-20, 20), repeats=1)

# [32.31389498710632, 41.10395312309265, 35.29583787918091,
# 41.88293409347534, 40.301613092422485, 41.43333315849304,
# 42.38564085960388, 42.07051205635071, 34.77747702598572,
# 47.61363887786865]

samples = NestedSamples(root='./chains/power-posterior')
fig, ax = samples.plot_2d([0, 1], label='power-posterior')
new_samples = NestedSamples(root='./chains/uniform')
new_samples.plot_2d(ax, label='uniform', alpha=0.5)
