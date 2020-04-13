from gaussian_models.parameter_covariance import ParameterCovarianceModel
from gaussian_models.power_posterior import log_likelihood_correction, power_gaussian_quantile


class GaussianPeakedPrior(ParameterCovarianceModel):
    """This is implemented as a special case of the power posterior repartitioning. One does need to care if it's the
    right function. """
    default_file_root = 'GaussianPosteriorModel'

    def log_likelihood(self, theta):
        log_l, phi = super().log_likelihood(theta)
        log_l += log_likelihood_correction(self, beta=1, theta=theta)
        return log_l, phi

    def quantile(self, cube):
        return power_gaussian_quantile(self, cube, 1)

    @property
    def dimensionality(self):
        return self.nDims
