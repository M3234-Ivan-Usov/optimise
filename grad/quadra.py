import matplotlib.pyplot as plt
from schedulers import *


class Quadratic:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.a = np.random.uniform(low=1.0, high=k, size=n)
        self.a[0], self.a[-1] = 1, k
        self.b = np.random.normal(loc=0, scale=1, size=n)

    def f(self, x):
        return 0.5 * np.dot(x * self.a, x) + np.dot(self.b, x)

    def df(self, x):
        return self.a * x + self.b

    @staticmethod
    def plot_quads(ns, ks, steps):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        k_mesh, n_mesh = np.meshgrid(ks, ns)
        ax.plot_surface(k_mesh, n_mesh, steps)
        ax.set_xlabel('K')
        ax.set_ylabel('N')
        ax.set_zlabel('Steps')
        plt.show()
