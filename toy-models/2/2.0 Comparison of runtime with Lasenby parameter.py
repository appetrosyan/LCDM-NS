import numpy
import pypolychord.priors
from pypolychord.settings import PolyChordSettings
import anesthetic
from scipy.special import erfinv
import time


# In this module I'm going to investigate the effect of the so-called
# Lasenby parameter as well as the effects of proper prior
# repartitioning, and benchmark to see a speedup.

# Vanilla

mu = numpy.array([1.0, 2.5])
cov = numpy.array([[1.0, 0.6], [0.6, 1.0]])
nDims = mu.size

settings = PolyChordSettings(nDims, 0)
settings.file_root = 'vanilla'
settings.nlive = 10**3
settings.read_resume = False
settings.do_clustering = True
settings.feedback = 0


def gaussian_likelihood(theta):
    invSig = numpy.linalg.inv(cov)
    norm = numpy.linalg.slogdet(2*numpy.pi * cov)[1]/2
    logL = - norm - (theta - mu)@ invSig @ (theta - mu) / 2
    return logL, []


def uniform_prior(point_in_hypercube):
    return pypolychord.priors.UniformPrior(-20, 20)(point_in_hypercube)


# try:
#     samples = anesthetic.NestedSamples(root='./chains/vanilla')
# except FileNotFoundError:
#     pypolychord.run_polychord(gaussian_likelihood,
#                               nDims, 0, settings, uniform_prior)
#     samples = anesthetic.NestedSamples(root='./chains/vanilla')


def bench(func, *args, repeats=1):
    times = []
    for i in range(repeats):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end-start)
    return times

# settings.file_root = 'vanilla'
# bench_results_vanilla = bench(pypolychord.run_polychord, gaussian_likelihood,
#                               nDims, 0, settings, uniform_prior, repeats=6)

# [29.408957958221436, 28.17777109146118, 27.804362058639526,
# 28.368597984313965, 27.895819902420044, 28.073410034179688]


L_matrix = numpy.linalg.cholesky(cov)
root2 = numpy.sqrt(2)


def gaussian_prior(point_in_hypercube):
    xinv = 2*point_in_hypercube - 1
    return mu + root2*L_matrix @ erfinv(xinv)


# settings.file_root = 'vanilla_gaussian'
# bench_results_vanilla_gaussian_prior = \
#    bench(pypolychord.run_polychord,
#          gaussian_likelihood, nDims, 0,
#          settings, gaussian_prior, repeats=6)


# [18.87489891052246, 18.819640159606934, 18.932925939559937,
# 19.270927667617798, 18.99773907661438, 19.10406994819641]


mu_lasenby = numpy.array([1.0, 2.5, 0.0])
cov_lasenby = numpy.array([[1.0, 0.6, 0],
                           [0.6, 1.0, 0.0],
                           [0.0, 0.0, 1]])


def gaussian_lasenby_likelihood(theta):
    invSig = numpy.linalg.inv(cov_lasenby)
    norm = numpy.linalg.slogdet(2*numpy.pi*cov_lasenby)[1]/2
    logL = - norm - (theta - mu_lasenby) @ invSig @ (theta - mu_lasenby) / 2
    return logL, []


L_matrix_lasenby = numpy.linalg.cholesky(numpy.linalg.inv(cov_lasenby))


def gaussian_lasenby_prior(point_in_hypercube):
    xinv = 2*point_in_hypercube - 1
    return mu_lasenby + root2+L_matrix_lasenby @ erfinv(xinv)


settings = pypolychord.settings.PolyChordSettings(3, 0)
settings.file_root = 'lasenby'
settings.read_resume = False
settings.nlive = 10**3
settings.do_clustering = True
settings.feedback = 0

bench_results_lasenby = bench(pypolychord.run_polychord,
                              gaussian_lasenby_likelihood, 3, 0,
                              settings, gaussian_lasenby_prior, repeats=6)

# [27.785404920578003, 33.215359926223755, 30.353167057037354,
# 30.423803091049194, 32.46933579444885, 30.452725172042847]


def temperature_repart_likelihood(theta):
    """\tilde{\pi} = \frac{\pi^\beta}{Z_\pi}"""
    beta = theta[-1:].item()
    tau = theta[:-1]
    x =
    return numpy.append(x, beta)
