from matplotlib import pyplot as plt
from numpy.random import uniform as initial
from schedulers import *


class Gladiator:
    def __init__(self, f, dim, df, scheduler):
        self.f, self.dim, self.df = f, dim, df
        self.scheduler = scheduler

    def __call__(self, iterations, eps, x0=None):
        if x0 is None:
            x0 = initial(-1 / self.dim, 1 / self.dim, self.dim)
        else:
            assert len(x0) == self.dim
        xs, fs, dfs = [x0], [self.f(x0)], [self.df(x0)]
        for it in range(1, iterations):
            xt, lr = xs[-1], self.scheduler(it, xs[-1], fs[-1], dfs[-1])
            xn = xt - lr * dfs[-1]
            if np.linalg.norm(xt - xn) < eps:
                break
            xs.append(xn), fs.append(self.f(xn)), dfs.append(self.df(xn))
        return xs, fs

    @staticmethod
    def plot_two_dim(args, values, fun, comment=""):
        xs, ys = [v[0] for v in args], [v[1] for v in args]
        xt, yt = (min(xs), max(xs)), (min(ys), max(ys))
        x_norm, y_norm = (xt[1] - xt[0]) * 0.05, (yt[1] - yt[0]) * 0.05
        xt = xt[0] - x_norm, xt[1] + x_norm
        yt = yt[0] - y_norm, yt[1] + y_norm
        x_lin = np.linspace(xt[0], xt[1], 500)
        y_lin = np.linspace(yt[0], yt[1], 500)
        x_m, y_m = np.meshgrid(x_lin, y_lin)
        f_m = fun([x_m, y_m])

        fig, ax = plt.subplots()
        ax.contourf(x_m, y_m, f_m, levels=81, cmap='magma')
        fig.set_figwidth(8)
        fig.set_figheight(8)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        x_, y_, f_ = xs[-1], ys[-1], values[-1]
        plt.plot(xs, ys, 'r.-')
        plt.plot(xs[0], ys[0], 'g.')
        plt.plot(xs[-1], ys[-1], 'gx')

        plt.title("steps: {}\nlast: {:.3f} ({:.3f}, {:.3f})\n{}".format(len(values), x_, y_, f_, comment))
        plt.show()
