import numpy as np
from matplotlib import pyplot as plt


class GoldenSearch:
    fi = (1 + np.sqrt(5)) / 2 - 1

    def __init__(self, f):
        self.f = f

    def __call__(self, left, right, eps=1e-6):
        x2 = left + self.fi * (right - left)
        x1 = right - self.fi * (right - left)
        fx1, fx2 = self.f(x1), self.f(x2)
        a, b, ps, fs = left, right, [], []
        while b - a > eps:
            ps.append((b + a) / 2)
            fs.append(self.f(ps[-1]))
            if fx1 < fx2:
                b, x2 = x2, x1
                x1 = b - self.fi * (b - a)
                fx2, fx1 = fx1, self.f(x1)
            else:
                a, x1 = x1, x2
                x2 = a + self.fi * (b - a)
                fx1, fx2 = fx2, self.f(x2)
        self.plot(left, right, ps, fs)
        return ps[-1], fs[-1]

    def plot(self, a, b, ps, fs):
        c = (b - a) * 0.05
        x_lin = np.linspace(a - c, b + c, 400)
        f_lin = self.f(x_lin)

        plt.title("Reached {:.3f} at {:.3f} by {} iterations".format(fs[-1], ps[-1], len(fs)))
        plt.plot(x_lin, f_lin)
        plt.plot(ps, fs, 'o')
        plt.plot(ps[-1], fs[-1], 'xr')
        plt.show()


def f1(x):
    return 3 * x ** 2 + 5 * x + 1


def f2(x):
    return 0.1 * x ** 4 - x ** 3 + 18 * x ** 2 - 2


fib1, fib2 = GoldenSearch(f2), GoldenSearch(f2)
fib1(-6, 5, 0.1)
