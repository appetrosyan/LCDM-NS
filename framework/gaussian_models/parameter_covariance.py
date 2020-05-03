from abc import ABC

from numpy import pi, array
from numpy.linalg import slogdet, multi_dot, inv, pinv

from polychord_model import Model


class ParameterCovarianceModel(Model, ABC):
    def __str__(self):
        return f'Gaussian posterior with \nmu = {self.mu}\ncov = {self.cov}'

    def __reprt__(self):
        return self.__str__().replace('\n', ' ')

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
            raise ValueError('Dimensions of cov and mean are incompatible: mean – {}, cov ({}, {}) '.format(
                self.nDims, rows, cols))
        try:
            self._invCov = inv(self.cov)
        except:
            print("Singular matrix, reverting to Penrose-Moore inverse, Singular Value Decomposition. ")
            self._invCov = pinv(self.cov)
        super().__init__(self.dimensionality, self.num_derived, file_root, **kwargs)

    def log_likelihood(self, theta):
        delta = theta - self.mu
        ll = - slogdet(2 * pi * self.cov)[1] / 2
        ll -= multi_dot([delta, self._invCov, delta]) / 2
        return ll, []
