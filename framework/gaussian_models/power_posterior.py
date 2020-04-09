from functools import lru_cache

from numpy import pi, log, array, concatenate, diag, sqrt, nextafter
from scipy.special import erf, erfinv

from gaussian_models.parameter_covariance import ParameterCovarianceModel


class PowerPosteriorPrior(ParameterCovarianceModel):
    default_file_root = 'PowerPosteriorModel'
    beta_min, beta_max = (nextafter(0, 1), 1)  # Smallest representable +ve float64

    def log_likelihood(self, theta):
        t = theta[:self.nDims]
        beta = theta[-1]
        logl, phi = super().log_likelihood(t)
        logl += log_likelihood_correction(self, beta, t)
        return logl, phi

    def quantile(self, cube):
        beta = self.beta_min + (self.beta_max - self.beta_min) * cube[-1]
        theta = power_gaussian_quantile(self, cube[:self.nDims], beta)
        return concatenate([theta, [beta]])

    @property
    def dimensionality(self):
        return self.nDims + 1


def _erf_term(d, b, g):
    @lru_cache(maxsize=2)
    def helper(t_delta, t_beta, t_sigma):
        hd, hg = array(t_delta), array(t_sigma)
        return erf(hd * sqrt(t_beta / 2) / hg)

    return helper(tuple(d), b, tuple(g))


def power_gaussian_quantile(m, cube, beta=1):
    sigma = diag(m.cov)
    da = _erf_term(m.a - m.mu, beta, sigma)
    db = _erf_term(m.b - m.mu, beta, sigma)
    ret = erfinv((1 - cube) * da + cube * db)
    return m.mu + sqrt(2 / beta) * sigma * ret


def log_box(m):
    if hasattr(m.b, '__iter__') or hasattr(m.a, '__iter__'):
        return log(m.b - m.a).sum()
    else:
        return m.nDims * log(m.b - m.a)


def log_likelihood_correction(model, beta, theta):
    ll = 0

    def ln_z(m, t, b):
        sigma = diag(m.cov)
        ret = - b * (t - m.mu) ** 2 / 2 / sigma ** 2
        ret -= log(pi * sigma ** 2 / 2 / b) / 2
        db = _erf_term(m.b - m.mu, b, sigma)
        da = _erf_term(m.a - m.mu, b, sigma)
        ret -= log(db - da)
        return ret

    ll -= log_box(model)
    ll -= ln_z(model, theta, beta).sum()

    return ll
