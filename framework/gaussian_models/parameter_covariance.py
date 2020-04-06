from numpy import pi, log, zeros, array
from numpy.linalg import slogdet, multi_dot, inv
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings

from polychord_model import Model


class ParameterCovarianceModel(Model):
    def __init__(self, bounds, mu, cov, file_root='paramCovModel', **kwargs):
        self.a, self.b = bounds
        self.mu = mu
        self.cov = cov
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
            raise ValueError('Dimensions of cov and mu are incompatible: mu – {}, cov ({}, {}) '.format(
                self.nDims, rows, cols))
        super().__init__(self.dimensionality, self.num_derived, file_root, **kwargs)

    def log_likelihood(self, theta):
        delta = theta - self.mu
        ll = - slogdet(2 * pi * self.cov)[1] / 2
        ll -= multi_dot([delta, inv(self.cov), delta]) / 2
        return ll, []
