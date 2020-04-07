from numpy import pi, log, zeros, array, nextafter, concatenate
from numpy.linalg import slogdet, multi_dot, inv
from pypolychord.priors import UniformPrior

from gaussian_models.parameter_covariance import ParameterCovarianceModel


class BoxUniformModel(ParameterCovarianceModel):
    default_file_root = 'boxUniform'

    def quantile(self, hypercube):
        return UniformPrior(self.a, self.b)(hypercube)


class StrawManResizeablePrior(ParameterCovarianceModel):
    """THis is an illustration of how not to do posterior repartitioning. """
    default_file_root = 'ResizeableBoxUniform'
    betamin = nextafter(0, 1)  # Smallest representable +ve float
    betamax = 1

    def log_likelihood(self, theta):
        t = theta[:self.nDims]
        beta = theta[-1]
        if beta <= self.betamin:
            beta = self.betamin
        logl, phi = super().log_likelihood(t / beta)
        logl += 2*self.nDims*(log(beta))
        return logl, phi

    def quantile(self, hypercube):
        beta = hypercube[-1:].item()
        # PolyChord refers to Quantile functions as priors. They are related, but not the same.
        uniform = UniformPrior(self.a*beta, self.b*beta)(hypercube[:-1])
        return concatenate([uniform, [beta]])

    @property
    def dimensionality(self):
        return self.nDims+1
