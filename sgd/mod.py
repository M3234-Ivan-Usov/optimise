import numpy as np


class Grad:
    def __init__(self, f, df, dim, lr=0.01):
        self.f, self.df = f, df
        self.lr, self.dim = lr, dim

    def __call__(self, xs, ys, batch_size, f_err, steps, init=None):
        initial = np.random.uniform(-1, 1, size=self.dim) if init is None else init
        approx, step = [initial], 0
        while True:
            for batch_idx in range(len(xs) // batch_size):
                start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
                x_values, y_actual = xs[start:end], ys[start:end]
                err = f_err(self.f(approx[-1], x_values), y_actual)
                grad = self.df(approx[-1], x_values, err)
                w = self.apply(grad)
                approx.append(approx[-1] - self.lr * w)
                if step == steps:
                    return approx
                else:
                    step += 1

    def apply(self, grad):
        return grad

    @staticmethod
    def sqr_err(predicted, actual):
        return predicted - actual


class RMSProp(Grad):
    def __init__(self, f, df, dim, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(f, df, dim, lr)
        self.beta, self.eps, self.sqr = beta, eps, np.zeros(dim)

    def apply(self, grad):
        self.sqr = self.beta * self.sqr + (1.0 - self.beta) * grad * grad
        return grad / (np.sqrt(self.sqr + self.eps))


class AdaDelta(Grad):
    def __init__(self, f, df, dim, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(f, df, dim, lr)
        self.delta, self.sqr = np.ones(dim), np.zeros(dim)
        self.beta, self.eps = beta, eps

    def apply(self, grad):
        self.sqr = self.beta * self.sqr + (1.0 - self.beta) * grad * grad
        w_update = np.sqrt(self.delta + self.eps) * grad / np.sqrt(self.sqr + self.eps)
        self.delta = self.beta * self.delta + (1.0 - self.beta) * w_update * w_update
        return w_update


class Adam(Grad):
    def __init__(self, f, df, dim, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(f, df, dim, lr)
        self.m, self.v = np.zeros(dim), np.zeros(dim)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.beta_pow1, self.beta_pow2 = 1.0, 1.0

    def apply(self, grad):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        self.beta_pow1 *= self.beta1
        self.beta_pow2 *= self.beta2
        m_norm = self.m / (1.0 - self.beta_pow1)
        v_norm = self.v / (1.0 - self.beta_pow2)
        return m_norm / (np.sqrt(v_norm) + self.eps)

