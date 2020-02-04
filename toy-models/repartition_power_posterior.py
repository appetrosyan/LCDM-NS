import numpy
from numpy import sqrt
import anesthetic
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from scipy.special import erf, erfinv

mu = numpy.array([1.0, 2.0, 3.0])
cov = numpy.array([[1, 0.5, 0.5],
                   [0.5, 2, 0.5],
                   [0.5, 0.5, 3]])

mu_pi = mu
sig_prior = numpy.sqrt(numpy.diagonal(cov))
a = numpy.array([-20.0, -20.0, -20.0])
b = numpy.array([20.0, 20.0, 20.0])
root2 = sqrt(2)
nDims = len(mu)                 # remember to pass nDims+1


def zed(beta):
    rr = sqrt(beta/2)
    arg1 = numpy.linalg.norm(rr*(b - mu_pi)/sig_prior)
    arg2 = numpy.linalg.norm(rr*(a - mu_pi)/sig_prior)
    # While analytically correct, the below result almost always fails
    # return (1/2)*(erf(arg1) - erf(arg2))


def log_zed(beta):
    rr = sqrt(beta/2)
    arg1 = numpy.linalg.norm(rr*(b - mu_pi)/sig_prior)
    arg2 = numpy.linalg.norm(rr*(a - mu_pi)/sig_prior)


def prior(cube):
    x = cube[:-1]
    beta = cube[-1:].item()
    rr = sqrt(beta/2)
    prior = mu_pi + root2 * \
        sig_prior * erfinv(
            x * erf(rr*(b-mu_pi) / sig_prior) +
            (1-x) * erf(rr*(a-mu_pi) / sig_prior)
        )/sqrt(beta)
    return prior


def log_likelihood(theta):
    x = theta[:-1]
    beta = theta[-1:].item()
    # norm =
    norm = numpy.log(numpy.product(numpy.diagonal(cov)))
    gaussian = -norm - ((x - mu_pi)/sig_prior)**2/2
    gaussian = numpy.sum(gaussian)
    return (2-beta)*(gaussian) + numpy.log(zed(beta)), []


settings = PolyChordSettings(nDims+1, 0)
settings.file_root = 'ppr'
settings.nlive = 20
# settings.do_clustering = True
# settings.read_resume = False

# run_polychord(log_likelihood, nDims+1, 0, settings, prior)
