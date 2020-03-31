
import copy
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from numpy import log, zeros
from anesthetic import NestedSamples


class Model:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def loglikelihood(self, theta):
        """A Loglikelihood respective of the model in question. This is the
unmodified version, i.e. without any kind of repartitioning.

        """
        raise NotImplementedError()

    def prior_inversion_function(self, hypercube):
        """Prior inversion function which maps the values in a unit hypercube
onto the real parameter space."""
        raise NotImplementedError()

    def eff_nDims(self):
        return self.nDims

    def _test_loglike(self):
        q, r = self.loglikelihood(zeros(self.eff_nDims()))

    def _test_prior(self):
        _nDims = len(self.prior_inversion_function(zeros(self.eff_nDims())))
        if _nDims != self.eff_nDims():
            raise ValueError(
                "Prior has the wrong dimensions: {} vs {} ".format(_nDims, self.eff_nDims()))

    def exec_polychord(self, verbosity=0, file_root=None, noResume=False, nLive=175):
        self._test_loglike()
        self._test_prior()
        _settings = copy.deepcopy(self.settings)
        _settings.feedback = verbosity
        if file_root is not None:
            _settings.file_root = file_root
        if noResume:
            _settings.read_resume = False
        _settings.nlive = nLive
        output = run_polychord(self.loglikelihood, self.eff_nDims(), self.nDerived,
                               _settings, self.prior_inversion_function)
        samples = NestedSamples(
            root='./chains/{}'.format(self.settings.file_root))
        return output, samples
