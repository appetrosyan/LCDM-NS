import sys
from time import time

import numpy
from anesthetic import NestedSamples
from pypolychord import run_polychord
from pypolychord.settings import PolyChordSettings
from scipy.special import erfinv

from DataGenerator import mixture_model


def model_linear(x, theta):
    m, c = theta
    return m * x + c


def model_quadratic(x, theta):
    m, c = theta
    return m * x ** 2 + c


# Start new derivation here.
def lnz(data_x, data_y, theta, model=model_linear, y_error=None):
    # http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    if y_error is None:
        y_error = numpy.ones(len(data_x), dtype=data_x.dtype)
    y = model(data_x, theta)
    chi_squared = numpy.sum(((data_y - y) / y_error) ** 2)
    log_z = -chi_squared / 2
    log_z -= len(data_x) / 2. * numpy.log(2 * numpy.pi)
    return log_z, []


# fn: [0,1] -> domain of parameter space.
def uniform_prior(theta, m_range=(-3, 3), c_range=(-3, 3)):
    min_m, max_m = m_range
    min_c, max_c = c_range
    tm, tc = theta
    m = min_m + tm * (max_m - min_m)
    c = min_c + tc * (max_c - min_c)
    return numpy.array([m, c])


def uncorrelated_gaussian_prior(theta, mu=(1.0, 1.0), sigma=(0.2, 0.8)):
    m = (mu[0] + erfinv(theta[0])) * sigma[0]
    c = (mu[1] + erfinv(theta[1])) * sigma[1]
    return numpy.array([m, c])


def run_iteration(file_root, prior, log_likelihood, nDims=2, nDerived=0):
    global settings
    settings.file_root = file_root
    settings.nDims = nDims
    settings.nDerived = nDerived
    settings.feedback = 0
    run_polychord(log_likelihood, nDims, nDerived, settings, prior)
    _samples = NestedSamples(root='chains/' + file_root)
    _samples.rename(columns={0: 'm', 1: 'c'}, inplace=True)
    return _samples


def bench(prior, series):
    # global linear_start, samples, linear_stop
    start = time()
    _samples = run_iteration(series, prior, log_likelihood)
    stop = time()
    return start, stop, _samples


if __name__ == "__main__":
    settings = PolyChordSettings(2, 0, read_resume=False)
    x_data = numpy.linspace(-3, 3, 250)
    # x_data, y_data = generate_noisy_from_model(lambda x: 1.2 * x + 1.2,
    # x_data, 0, 0.1)
    x_data, y_data = mixture_model(lambda x: 1.1 * x + 1.1,
                                   lambda x: 1.4 * x + 1.1,
                                   x_data, x_noise_amplitude=0,
                                   y_noise_amplitude=0.1)
    log_likelihood = lambda theta: lnz(x_data, y_data, theta)

    linear_start, linear_stop, samples = bench(uniform_prior, 'linear')
    # samples.plot_2d(['m', 'c'])
    # plt.show()

    muc = numpy.linspace(0.1, 2, 10)
    mum = numpy.linspace(0.1, 2, 10)
    sigmam = numpy.linspace(0.01, 2, 10)
    sigmac = numpy.linspace(0.01, 2, 10)
    print("sc,\tsm,\tmu_c,\tmu_m,\tlnZ,\tD,\td ", file=sys.stderr)
    for mu_m in mum:
        for mu_c in muc:
            for sigma_m in sigmam:
                for sigma_c in sigmac:
                    p = lambda x: uncorrelated_gaussian_prior(x, (mu_m, mu_c),
                                                              (sigma_m, sigma_c))
                    print('{:.4},\t{:.4},\t{:.4},\t{:.4},\t'
                          .format(sigma_c, sigma_m, mu_c, mu_m),
                          file=sys.stderr,
                          end='')
                    gaussian_start, gaussian_stop, samples =
                    bench(p, 'gaussian')
                    print('{:.4},\t{:.4},\t{:.4}'.format(samples.logZ(),
                                                         samples.D(),
                                                         samples.d()),
                          file=sys.stderr)
    # samples.plot_2d(['m', 'c'])
    # plt.show()

    # print("\n\n\n\n\n\n\n{} ".format(linear_start - linear_stop, )#gaussian_start - gaussian_stop))
