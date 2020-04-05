
import copy
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from numpy import log, zeros
from anesthetic import NestedSamples


class Model:
    default_file_root = 'blankModel'

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def loglikelihood(self, theta):
        """A Ln(likelihood) of the data given the model. With a uniform
        prior, this defines the posterior distribution up to a
        multiplicative factor.

        """
        raise NotImplementedError()

    def prior_inversion_function(self, hypercube):
        """Inverse Cumulative distribution function of the prior. Aka the
        quantile. If the prior has PDF \pi, then this is (CDF (pi))^-1. 

        """
        raise NotImplementedError()

    def eff_nDims(self):
        """
        This is how many elements in the \theta vector need to be
        present. Usually the same as the nDims of the nested self.settings
        object.
        """
        return self.nDims

    def _test_loglike(self):
        p = self.loglikelihood(zeros(self.eff_nDims()))
        try:
            q, r = p
        except ValueError as e:
            raise ValueError(
                e.msg + "Did you forget to return the derived parameters?")

    def _test_prior(self):
        _nDims = len(self.prior_inversion_function(zeros(self.eff_nDims())))
        if _nDims != self.eff_nDims():
            raise ValueError(
                "Prior has the wrong dimensions: expect {} vs actual {} ".format(_nDims, self.eff_nDims()))

    def exec_polychord(self, verbosity=0, file_root=None, noResume=False, nLive=175):
        self._test_loglike()
        self._test_prior()
        _settings = copy.deepcopy(self.settings)
        _settings.feedback = verbosity
        if not verbosity == 0:
            print('output set to verbose')
        if file_root is not None:
            _settings.file_root = file_root
        if noResume:
            _settings.read_resume = False
        _settings.nlive = nLive
        output = run_polychord(self.loglikelihood, self.eff_nDims(), self.nDerived,
                               _settings, self.prior_inversion_function)
        try:
            samples = NestedSamples(
                root='./chains/{}'.format(_settings.file_root))
        except ValueError as e:
            print(e)
            samples = None
        return output, samples
