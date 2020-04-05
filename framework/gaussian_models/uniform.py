from gaussian_models.parameter_covariance import ParameterCovarianceModel
from pypolychord.priors import UniformPrior
from numpy.linalg import slogdet, multi_dot, inv
from numpy import pi, log, zeros, array, nextafter, concatenate


class BoxUniformModel(ParameterCovarianceModel):
    default_file_root = 'boxUniform'

    def prior_inversion_function(self, hypercube):
        return UniformPrior(self.a, self.b)(hypercube)


class ResizeableUniformPrior(ParameterCovarianceModel):
    default_file_root = 'ResizeableBoxUniform'
    betamin = nextafter(0, 1)  # Smallest representable +ve float
    betamax = 1

    def loglikelihood(self, theta):
        t = theta[:self.nDims]
        beta = theta[-1]
        if beta <= self.betamin:
            beta = self.betamin
        logl, phi = super().loglikelihood(t/beta)
        logl += 2*self.nDims*(log(beta))
        return logl, phi

    def prior_inversion_function(self, hypercube):
        b = hypercube[-1:].item()
        # Probably very unnecessary but it works for now.
        up = UniformPrior(self.a*b, self.b*b)(hypercube[:-1])
        return concatenate([up, [b]])

    def eff_nDims(self):
        return self.nDims+1
