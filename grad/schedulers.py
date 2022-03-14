from abc import ABC, abstractmethod
import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass


class ConstScheduler(Scheduler):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, *args):
        return self.lr


class ExpScheduler(Scheduler):
    def __init__(self, lr0, k=0.001):
        self.lr0, self.k = lr0, k

    def __call__(self, it, *args):
        return self.lr0 * np.exp(it * -self.k)


class ArmijoScheduler(Scheduler):
    def __init__(self, f, df, alpha_space, c1=1e-4, c2=0.9):
        self.f, self.df = f, df
        self.alphas = alpha_space
        self.c1, self.c2 = c1, c2

    def wolfe1(self, x, fx, dfx, alpha, grad_norm):
        return self.f(x - alpha * dfx) - fx + self.c1 * alpha * grad_norm

    def wolfe2(self, x, fx, dfx, alpha, grad_norm):
        return np.dot(dfx, self.df(x - alpha * dfx)) - self.c2 * grad_norm

    def __call__(self, it, x, fx, dfx):
        grad_norm, wolves = np.dot(dfx, -dfx), []
        for alpha in self.alphas:
            w1 = self.wolfe1(x, fx, dfx, alpha, grad_norm)
            if w1 >= 0:
                continue
            w2 = self.wolfe2(x, fx, dfx, alpha, grad_norm)
            if w1 < 0:
                wolves.append((w1, w2, alpha))
        if len(wolves) == 0:
            return self.alphas[0]
        wolfe_alphas = list(filter(lambda w: w[1] < 0, wolves))
        if len(wolfe_alphas) != 0:
            return min(wolfe_alphas, key=lambda w: w[0])[2]
        return min(wolves, key=lambda w: w[0])[2]

