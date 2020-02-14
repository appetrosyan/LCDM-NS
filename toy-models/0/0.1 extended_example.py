import numpy as np
import matplotlib.pyplot as plt
import pypolychord as ppc
import pypolychord.settings
# import anesthetic
import argparse
# import time


def f(x, theta):
    """"A function used for line fitting. In this case Linear"""
    m, c = theta
    return m*x + c


def lnL(theta):
    """Logarithm of the likelihood"""
    y = f(dx, theta)
    chisq = np.sum(((y-dy)/derr)**2)
    # TODO: What to do about derived parameters?
    return (-chisq/2) - len(dx)/2.*np.log(2*np.pi), []


def prior(cube, m_range=(-10, 10), c_range=(-10, 10)):
    """Prior generated in a default range of (-3, 3)"""
    m = m_range[0] + cube[0] * (m_range[1] - m_range[0])
    c = c_range[0] + cube[1] * (c_range[1] - c_range[0])
    return np.array([m, c])


def symmetric_unfirom_prior(cube, m=20, c=20):
    return prior(cube, (-m, m), (-c, c))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file with data.")
    parser.add_argument("--silent",
                        help="suppress production of plots",
                        action='store_true')
    args = parser.parse_args()
    print("loading {}".format(args.file))
    if args.file:
        dx, dy, derr = np.loadtxt(vars(args)['file']).T
        settings = ppc.settings.PolyChordSettings(2, 0)
        settings.file_root = 'linear'  # Prefix name
        settings.read_resume = False   # Don't re-read and resume
        if args.silent:
            settings.feedback = 0
        q = []
        for mc in range(50):
            print("{:.6}s\t {}".format(q[mc-1], mc))

        plt.plot(range(50), q, fmt='k+')
        samples.rename(columns={0: 'm', 1: 'c'}, inplace=True)
        if not args.silent:
            samples.plot_1d(['m', 'c'])
            plt.show()
