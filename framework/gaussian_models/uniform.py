from numpy import log, nextafter, concatenate
from pypolychord.priors import UniformPrior

from gaussian_models.parameter_covariance import ParameterCovarianceModel


class BoxUniformModel(ParameterCovarianceModel):
    default_file_root = 'boxUniform'

    def quantile(self, hypercube):
        return UniformPrior(self.a, self.b)(hypercube)


class StrawManResizeablePrior(ParameterCovarianceModel):
    """THis is an illustration of how not to do posterior repartitioning. """
    default_file_root = 'ResizeableBoxUniform'
    beta_min = nextafter(0, 1)  # Smallest representable +ve float
    beta_max = 1

    def log_likelihood(self, theta):
        t = theta[:self.nDims]
        beta = theta[-1]
        if beta <= self.beta_min:
            beta = self.beta_min
        log_l, phi = super().log_likelihood(t / beta)
        log_l += 2 * self.nDims * (log(beta))
        # Uncomment this to fix the repartitioning.
        # import gaussian_models.power_posterior
        # log_l -= gaussian_models.power_posterior.log_box(self)
        return log_l, phi

    def quantile(self, hypercube):
        beta = hypercube[-1:].item()
        # PolyChord refers to Quantile functions as priors.
        # This is not incorrect, but can be confusing.
        uniform = UniformPrior(self.a * beta, self.b * beta)(hypercube[:-1])
        return concatenate([uniform, [beta]])

    @property
    def dimensionality(self):
        return self.nDims + 1
