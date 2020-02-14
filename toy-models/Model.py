#!/usr/bin/env python
import numpy
from numpy import log, product, diagonal, pi, sqrt
from pypolychord import run_polychord
import anesthetic
from pypolychord.priors import UniformPrior
from pypolychord.settings import PolyChordSettings
from scipy.special import erf, erfinv


class CorrelatedGaussianModel(object):
    def __init__(self, mu, cov, file_root='untitled'):
        self.mu = mu if isinstance(mu, numpy.ndarray) else numpy.array(mu)
        self.cov = cov if isinstance(cov, numpy.ndarray) else numpy.array(cov)
        self.isDiagonal = False
        self.nDims = self.mu.size
        shape = self.cov.shape
        if(len(shape) > 2):
            raise ValueError(
                "Covariance matrix has higher rank than a 2-tensor. {}".
                format(shape))
        if len(shape) == 2:
            a, b = shape
            if a != b:
                raise ValueError(
                    "Covariance matrix not square. It has shape {}x{}".
                    format(a, b))
            if a != self.mu.size:
                raise ValueError("Covariance matrix has different dimesnions to mean vector. {}x{} vs {}".
                                 format(a, b, self.mu.size))
        else:
            if self.cov.size != self.mu.size:
                raise ValueError(
                    "Incompatible standard deviation vector. [sigma]={}, [mu]={}".
                    format(self.cov.size, self.mu.size))
            else:
                self.isDiagonal = True
        self.__invCov = numpy.linalg.inv(
            self.cov) if not self.isDiagonal else numpy.linalg.inv(numpy.diag(self.cov))
        self.settings = PolyChordSettings(self.nDims, 0)
        self.settings.file_root = file_root
        self.settings.read_resume = False
        self.settings.do_clustering = True
        self.settings.nlive = 20

    def logL_(self, theta):
        norm = numpy.linalg.slogdet(2*numpy.pi*self.cov)[1]/2
        return -norm - \
            numpy.linalg.multi_dot(
                [theta - self.mu, numpy.linalg.inv(self.cov), theta - self.mu])/2, []

    def prior(self, cube):
        return UniformPrior(-20, 20)(cube)

    def run_pc(self, nlive=20):
        run_polychord(lambda x: self.logL_(x), self.nDims, 0,
                      self.settings, lambda x: self.prior(x))


a = CorrelatedGaussianModel([1, 2, 3], numpy.diag([1, 2, 3]))


class UncorrelatedGaussianModel(object):
    def __init__(self, mu, sigma, file_root='untitled'):
        self.mu = mu
        if not isinstance(self.mu, numpy.ndarray):
            self.mu = numpy.array(self.mu)
        self.sigma = sigma
        if not isinstance(self.sigma, numpy.ndarray):
            self.sigma = numpy.array(self.sigma)
        if len(self.sigma.shape) != 1:
            raise ValueError(
                "Standard deviations not provided as a vector, maybe you meant to use CorrelatedGaussianModel? {}".
                format(sigma))
        else:
            if self.mu.size != self.sigma.size:
                raise ValueError("Incompatible sizes of standard deviation and mean vectors. [mu]={} vs [sigma]={}".
                                 format(self.mu.size, self.sigma.size))
        self.nDims = self.mu.size
        self.settings = Polychordsettings(self.nDims, 0)
        self.settings.file_root = file_root
        self.settings.read_resume = False
        self.settings.do_clustering = True
        self.settings.nlive = 20

        def logL(self, theta):
            norm = numpy.sum(log(2*pi*sigma))/2
            gaussian = -norm - numpy.sum(((x - self.mu)/self.sigma)**2)/2
            return gaussian, []

        def prior(self, cube):
            return UniformPrior(-20, 20)(cube)

        def run_pc(self, nlive=20):
            run_polychord(lambda x: self.logL(x), self.nDims, 0,
                          self.settings, lambda x: self.prior(x))


class RepartitionedGaussianModel(UncorrelatedGaussianModel):
    def __init__(self, mu, sigma, b, a, file_root=''):
        # super(RepartitionedGaussianModel, self).__init__(self, mu, cov)
        self.mu = mu
        if not isinstance(self.mu, numpy.ndarray):
            self.mu = numpy.array(self.mu)
        self.sigma = sigma
        if not isinstance(self.sigma, numpy.ndarray):
            self.sigma = numpy.array(self.sigma)
        if len(self.sigma.shape) != 1:
            raise ValueError(
                "Standard deviations not provided as a vector, maybe you meant to use CorrelatedGaussianModel? {}".
                format(sigma))
        else:
            if self.mu.size != self.sigma.size:
                raise ValueError("Incompatible sizes of standard deviation and mean vectors. [mu]={} vs [sigma]={}".
                                 format(self.mu.size, self.sigma.size))
        self.nDims = self.mu.size
        self.settings = PolyChordSettings(self.nDims, 0)
        self.settings.file_root = file_root
        self.settings.read_resume = False
        self.settings.do_clustering = True
        self.settings.nlive = 20
        self.upper_bounds = b
        self.lower_bounds = a

    def zed(self, beta):
        root2 = sqrt(beta/2)
        arg1 = numpy.linalg.norm(
            root2*(self.upper_bounds - self.mu)/self.sigma)
        arg2 = numpy.linalg.norm(
            root2*(self.lower_bounds - self.mu)/self.sigma)
        # While analytically correct, the below result almost always fails
        return (1/2)*(erf(arg2) - erf(arg1))

    def _log_zed(self, beta):
        rr = sqrt(beta/2)
        arg1 = numpy.linalg.norm(rr*(self.upper_bounds - self.mu)/self.sigma)
        arg2 = numpy.linalg.norm(rr*(self.lower_bounds - self.mu)/self.sigma)
        if arg1 == arg2:
            return 0
        else:
            if (arg1 > 6 and arg2 > 6) or (arg1 < -6 and arg2 < -6):
                derivative = numpy.exp(-arg1**2)*(1/sqrt(numpy.pi))
                diff = derivative*(arg2 - arg1)  # Very small
                return log(arg2-arg1) - (arg1)**2 - log(numpy.pi)/2
            else:
                return log(erf(arg2) - erf(arg1)) - log(2)

    def logL(self, theta):
        x = theta[:-1]
        beta = theta[-1:].item()

        norm = numpy.sum(log(2*pi*sigma))/2
        gaussian = -norm - numpy.sum(((x - self.mu)/self.sigma)**2)/2
        return super.logL + (1-beta)*gaussian + _log_zed(beta)

    def inv_prior(self, cube):
        x = cube[:-1]
        beta = cube[-1:].item()
        root_beta_half = sqrt(beta/2)
        prior = self.mu + root_beta_half * self.sigma * erfinv(x*erf(root_beta_half*(self.upper_bounds - self.mu) / self.sigma) + (
            1 - x)*erf(root_beta_half * (self.lower_bounds - self.mu)/self.sigma))/sqrt(beta)
        return numpy.append(prior, beta)
