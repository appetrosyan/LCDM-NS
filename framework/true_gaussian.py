from parameter_covariance import ParameterCovarianceModel
from numpy import log, pi, diag, sqrt, array, zeros
from numpy.linalg import inv, slogdet
from scipy.special import erf, erfinv
from pypolychord.priors import GaussianPrior


class GaussianPeakedPrior(ParameterCovarianceModel):
    default_file_root = 'GaussianPosteriorModel'

    def eff_ndims(self):
        return self.nDims

    def _ln_z(self, theta):
        sigma = diag(self.cov)
        ret = - (theta-self.mu)**2/2/sigma**2
        ret -= log(pi*sigma**2/2)/2
        ret -= log(erf((self.b-self.mu)*sqrt(1/2)/sigma)
                   - erf((self.a-self.mu)*sqrt(1/2)/sigma))
        return ret

    def prior_inversion_function(self, hypercube):
        return self._power_gaussian(hypercube)

    def _power_gaussian(self, hypercube):
        sigma = diag(self.cov)
        ret = erfinv((1-hypercube)*erf((self.a-self.mu)*sqrt(1/2)/sigma) +
                     hypercube*erf((self.b-self.mu)*sqrt(1/2)/sigma))
        return self.mu + sqrt(2)*sigma*ret

    def loglikelihood(self, theta):
        logl, phi = super().loglikelihood(theta)
        logl -= len(theta)*log(self.b - self.a)
        logl -= self._ln_z(theta).sum()
        return logl, phi
