from copy import deepcopy

from anesthetic import NestedSamples
from numpy import zeros

# As of now PolyChord is not pip-installable.
# noinspection PyUnresolvedReferences,PyUnresolvedReferences

from pypolychord import run_polychord
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from pypolychord.settings import PolyChordSettings


class Model:
    default_file_root = 'blankModel'

    def __init__(self, dimensionality, number_derived, file_root='', **kwargs):
        self.settings = PolyChordSettings(dimensionality, number_derived)
        self.settings.file_root = file_root

    def log_likelihood(self, theta):
        """A Ln(likelihood) of the runs given the model. With a uniform
        prior, this defines the posterior distribution up to a
        multiplicative factor.

        """
        raise NotImplementedError()

    def quantile(self, hypercube):
        """Inverse Cumulative distribution function of the prior. Aka the
        quantile. If the prior has PDF \\pi, then this is (CDF (\\pi))^-1.

        """
        raise NotImplementedError()

    @property
    def dimensionality(self):
        """
        This is how many elements in the \\theta vector need to be
        present. Usually the same as the nDims of the nested self.settings
        object. This is read-only. Ideally it should be cached.
        """
        return int(self.nDims)

    @property
    def num_derived(self):
        return 0

    def test_log_like(self):
        p = self.log_likelihood(zeros(self.dimensionality))
        try:
            _, _ = p
        except ValueError as e:
            raise ValueError(e.msg + "Did you forget to return the derived parameters?")

    def test_quantile(self):
        _nDims = len(self.quantile(zeros(self.dimensionality)))
        if _nDims != self.dimensionality:
            raise ValueError(f'Prior has the wrong dimensions: expect {_nDims} vs actual {self.dimensionality}')

    def nested_sample(self, **kwargs):
        self.test_log_like()
        self.test_quantile()
        _settings = self.setup_settings(**kwargs)
        output = run_polychord(self.log_likelihood, self.dimensionality, self.num_derived, _settings, self.quantile)
        try:
            samples = NestedSamples(
                root=f'./chains/{_settings.file_root}')
        except ValueError as e:
            print(e)
            samples = None
        return output, samples

    # noinspection SpellCheckingInspection
    def setup_settings(self, file_root=None, live_points=175, resume=True, verbosity=0):
        _settings = deepcopy(self.settings)
        _settings.feedback = verbosity
        if file_root is not None:
            _settings.file_root = file_root
        _settings.read_resume = resume
        _settings.nlive = live_points
        return _settings
