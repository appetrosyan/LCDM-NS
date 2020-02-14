import numpy
from numpy import sqrt
import anesthetic
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from scipy.special import erf, erfinv
import decimal

mu = numpy.array([1.0, 2.0, 3.0])
cov = numpy.array([[1, 0.0, 0.0],
                   [0.0, 2, 0.0],
                   [0.0, 0.0, 3]])

mu_pi = mu
sig_prior = numpy.sqrt(numpy.diagonal(cov))
a = numpy.array([-5.0, -5.0, -5.0])
b = numpy.array([5.0, 5.0, 5.0])
root2 = sqrt(2)
nDims = len(mu)                 # remember to pass nDims+1


def zed(beta):
    rr = sqrt(beta/2)
    arg1 = numpy.linalg.norm(rr*(b - mu_pi)/sig_prior)
    arg2 = numpy.linalg.norm(rr*(a - mu_pi)/sig_prior)
    # While analytically correct, the below result almost always fails
    return (1/2)*(erf(arg2) - erf(arg1))


def log_zed(beta):
    rr = sqrt(beta/2)
    arg1 = numpy.linalg.norm(rr*(b - mu_pi)/sig_prior)
    arg2 = numpy.linalg.norm(rr*(a - mu_pi)/sig_prior)
    if arg1 == arg2:
        return 0
    else:
        if (arg1 > 6 and arg2 > 6) or (arg1 < -6 and arg2 < -6):
            # Huge loss of precision. We need to expand to linear order
            derivative = numpy.exp(-arg1**2)*(1/sqrt(numpy.pi))
            diff = derivative*(arg2 - arg1)  # Very small
            # return numpy.log(diff)
            # return numpy.log(derivative)+numpy.log(arg2 - arg1)
            return numpy.log(arg2-arg1) - (arg1)**2 - numpy.log(numpy.pi)/2
        else:
            return numpy.log(erf(arg2) - erf(arg1)) - numpy.log(2)


1


def prior(cube):
    x = cube[:-1]
    beta = cube[-1:].item()
    rr = sqrt(beta/2)
    prior = mu_pi + root2 * \
        sig_prior * erfinv(
            x * erf(rr*(b-mu_pi) / sig_prior) +
            (1-x) * erf(rr*(a-mu_pi) / sig_prior)
        )/sqrt(beta)
    return numpy.append(prior, beta)
    # return UniformPrior(-5, 5)(cube)


def log_likelihood(theta):
    x = theta[:-1]
    beta = theta[-1:].item()
    # norm =
    norm = numpy.log(numpy.product(numpy.diagonal(numpy.pi*2*cov)))/2
    gaussian = -norm - numpy.sum(((x - mu_pi)/sig_prior)**2)/2
    # return (2-beta)*(gaussian) + numpy.log(zed(beta)), []
    # print(gaussian, log_zed(beta), beta)
    return (2-beta)*gaussian + log_zed(beta), []
    # return gaussian, []
    # return -numpy.linalg.slogdet(cov)[1]/2 - (x - mu_pi)@ numpy.linalg.inv(cov)@(x - mu_pi)/2, []


settings = PolyChordSettings(nDims+1, 0)
settings.file_root = 'ppr'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False


def exe():
    run_polychord(log_likelihood, nDims+1, 0, settings, prior)


samples = anesthetic.NestedSamples(root='./chains/ppr')
# floating point overflow:
# zed(0.5) = -inf
