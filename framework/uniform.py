from parameter_covariance import ParameterCovarianceModel
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from numpy.linalg import slogdet, multi_dot, inv
from numpy import pi, log, zeros, array


class BoxUniformModel(ParameterCovarianceModel):
    default_file_root = 'boxUniform'

    def prior_inversion_function(self, hypercube):
        return UniformPrior(self.a, self.b)(hypercube)
