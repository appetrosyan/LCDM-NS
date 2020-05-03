import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import common_dists as cd
import random


def stoch(x, y, beta=0.75):
    rval = []
    h = hash((x, y))
    random.seed(h)
    r = random.random()
    if r > beta:
        return cd.N(x, 0, 0.5)*cd.N(y, 0, 0.5)
    else:
        return cd.U(x, -1, 1)*cd.U(y, -1, 1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
y = np.arange(-0, 1.2, 2**(-5))
x = np.arange(-1.2, 1.2, 2**(-5))
X, Y = np.meshgrid(x, y)


def apply(fun, X, Y):
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    return Z


def plot_slice(beta):
    global cset
    Z = apply(lambda x, y: stoch(x, y, 1-beta), X, Y)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=beta,
                       levels=np.linspace(-0.2, 1.2, 1000), cmap=plt.cm.jet)


plot_slice(0)
plot_slice(0.25)
plot_slice(0.75)
plot_slice(1)
ax.set_xlabel('$\\theta_0$')
ax.set_ylabel('$\\theta_1$')
ax.set_zlabel('$\\beta$')
ax.set_zlim(0, 1.2)
cbar = fig.colorbar(cset, shrink=0.5, aspect=5,
                    label=r'$\pi(\mathbf{\theta})$', ticks=[0, 0.44, 0.66, 1])
cbar.ax.set_yticklabels([r'$0$', r'uniform', r'$max(\pi)$', r'$1$'])
plt.show()
