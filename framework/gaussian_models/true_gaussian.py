from numpy import log, pi, diag, sqrt, array, zeros
from numpy.linalg import inv, slogdet
from pypolychord.priors import GaussianPrior
from scipy.special import erf, erfinv

from gaussian_models.parameter_covariance import ParameterCovarianceModel

from gaussian_models.power_posterior import PowerPosteriorPrior, log_likelihood_correction, power_gaussian_quantile

class GaussianPeakedPrior(ParameterCovarianceModel):
    """THis is implemented as a special case of the power posterior repartitioning. One does need to care if it's the right function. """
    default_file_root = 'GaussianPosteriorModel'

    def log_likelihood(self, theta):
        logl, phi = super().log_likelihood(theta)
        logl += log_likelihood_correction(self, beta=1, theta=theta)
        return logl, phi

    def quantile(self, cube):
        return power_gaussian_quantile(self, cube, 1)

    @property
    def dimensionality(self):
        return self.nDims