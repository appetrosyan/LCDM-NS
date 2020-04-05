from parameter_covariance import ParameterCovarianceModel
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from numpy.linalg import slogdet, multi_dot, inv
from numpy import pi, log, zeros, array, concatenate, diag, sqrt, nextafter
from scipy.special import erf, erfinv


class PowerPosteriorPrior(ParameterCovarianceModel):
    default_file_root = 'PowerPosteriorModel'
    betamin = nextafter(0, 1)   # Smallest representable +ve float64
    betamax = 1

    def _ln_z(self, theta, beta):
        sigma = diag(self.cov)
        ret = -beta * (theta-self.mu)**2/2/sigma**2
        ret -= log(pi*sigma**2/2/beta)/2
        ret -= log(erf((self.b-self.mu)*sqrt(beta/2)/sigma)
                   - erf((self.a-self.mu)*sqrt(beta/2)/sigma))
        return ret

    def loglikelihood(self, theta):
        t = theta[:self.nDims]
        beta = theta[-1]
        logl, phi = super().loglikelihood(t)
        logl -= len(t) * log(self.b-self.a)
        logl -= self._ln_z(t, beta).sum()
        return logl, phi

    def _power_gaussian(self, hypercube, beta):
        sigma = diag(self.cov)
        ret = erfinv((1-hypercube)*erf((self.a-self.mu)*sqrt(beta/2)/sigma) +
                     hypercube*erf((self.b-self.mu)*sqrt(beta/2)/sigma))
        return self.mu + sqrt(2/beta)*sigma*ret

    def prior_inversion_function(self, cube):
        x = cube[:self.nDims]
        b = cube[-1]
        beta = self.betamin + (self.betamax - self.betamin)*b
        theta = self._power_gaussian(x, beta)
        return concatenate([theta, [beta]])

    def eff_nDims(self):
        return self.nDims + 1
