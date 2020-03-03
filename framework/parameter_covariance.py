from polychord_model import Model
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from numpy.linalg import slogdet, multi_dot, inv
from numpy import pi, log, zeros, array


class ParameterCovarianceModel(Model):
    default_file_root = 'parameterCovarianceModel'

    def __init__(self, bounds, mu, cov,
                 settings=None, file_root=default_file_root):
        self.a, self.b = bounds
        self.mu = mu
        self.cov = cov
        self.nDerived = 0
        try:
            self.nDims = mu.size
        except AttributeError:
            self.mu = array(mu)
            self.nDims = mu.size

        try:
            rows, cols = self.cov.shape
        except AttributeError:
            self.cov = array(self.cov)
            rows, cols = self.cov.shape  # raise VE if too few to unpack
            if rows != cols or rows != self.nDims:
                raise ValueError('dimesnions of cov and mu are incompatible: mu – {}, cov ({}, {}) '.format(
                    self.nDims, rows, cols))

        if settings is None:
            self.settings = PolyChordSettings(self.eff_nDims(), self.nDerived)
        else:
            self.settings = setting
        self.settings.file_root = file_root

    def loglikelihood(self, theta):
        delta = theta - self.mu
        ll = - slogdet(2*pi*self.cov)[1]/2
        ll -= multi_dot([delta, inv(self.cov), delta])/2
        return ll, []
