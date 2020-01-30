
import Sig
import numpy
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

nDims = 2
nDerived = 0

mu = numpy.array([0,0])
Sig = numpy.array([[1,0.99],[0.99,1]])
invSig = numpy.linalg.inv(Sig)
norm = numpy.linalg.slogdet(2*numpy.pi*Sig)[1]/2


def likelihood(theta):
    """ Simple Gaussian Likelihood"""
    logL = - norm  - (theta - mu) @ invSig @ (theta - mu) / 2

    #logL -= stuff that is related to the beta parameter
    return logL, []


def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(-20, 20)(hypercube)


settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'gaussian_2'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
from anesthetic import NestedSamples
samples = NestedSamples(root='./chains/gaussian')
fig, ax = samples.plot_2d([0,1])
samples_1 = NestedSamples(root='./chains/gaussian_1')
samples_1.plot_2d(ax)
samples_2 = NestedSamples(root='./chains/gaussian_2')
samples_2.plot_2d(ax)



samples = NestedSamples(root='/data/will/tension/runs/lcdm/chains/planck')
samples.plot_2d(['logA', 'ns'])

params = samples.columns[:27]
Sig = samples[params].cov().values
mu = samples[params].mean().values
invSig = numpy.linalg.inv(Sig)
norm = numpy.linalg.slogdet(2*numpy.pi*Sig)[1]/2
nDims = len(mu)

planck_ranges = numpy.array(
            [[0.019,0.025],
             [0.095,0.145],
             [1.03,1.05],
             [0.01,0.4],
             [2.5,3.7],
             [0.885,1.04],
             [0.9,1.1],
             [0,200],
             [0,1],
             [0,10],
             [0,400],
             [0,400],
             [0,400],
             [0,400],
             [0,10],
             [0,50],
             [0,50],
             [0,100],
             [0,400],
             [0,10],
             [0,10],
             [0,10],
             [0,10],
             [0,10],
             [0,10],
             [0,3],
             [0,3]])

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return planck_ranges[:,0] + hypercube * (planck_ranges[:,1]- planck_ranges[:,0])


settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'gaussian_2'
settings.nlive = 200
settings.do_clustering = True
settings.read_resume = False

samples = NestedSamples(root='/data/will/tension/runs/lcdm/chains/planck')
samples.plot_2d(['logA', 'ns'])

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior)
