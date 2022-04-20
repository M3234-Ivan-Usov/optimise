import math

import numpy as np
from matplotlib import pyplot as plt
from mod import *
import profile


def f_lin(p, x):
    return p[0] * x + p[1]


def df_lin(p, x, err):
    return np.array([np.mean(x * err), np.mean(err)])


def generate_lin(n, sigma, x_range=(-1, 1), param_range=(-1, 1)):
    p_a = np.random.uniform(low=param_range[0], high=param_range[1])
    p_b = np.random.uniform(low=param_range[0], high=param_range[1])
    error = np.random.normal(loc=0.0, scale=sigma, size=n)
    xs = np.random.uniform(low=x_range[0], high=x_range[1], size=n)
    return p_a, p_b, xs, f_lin(np.array([p_a, p_b]), xs) + error


COLORS = ['aqua', 'darkblue', 'lightgreen', 'fuchsia', 'violet']


def plot_lin(xs, ys, approx, params, real):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(7)

    a_approx = [extract_var(app, 0) for app in approx]
    b_approx = [extract_var(app, 1) for app in approx]

    a_min, a_max, b_min, b_max = real[0], real[0], real[1], real[1]
    for app_idx in range(len(a_approx)):
        a_min = min(a_min, min(a_approx[app_idx]))
        a_max = max(a_max, max(a_approx[app_idx]))
        b_min = min(b_min, min(b_approx[app_idx]))
        b_max = max(b_max, max(b_approx[app_idx]))
    a_extra, b_extra = 0.05 * (a_max - a_min), 0.05 * (b_max - b_min)

    a_lin = np.linspace(a_min - a_extra, a_max + a_extra, 120)
    b_lin = np.linspace(b_min - b_extra, b_max + b_extra, 120)
    a_space, b_space, x_space = np.meshgrid(a_lin, b_lin, xs)
    y_space = f_lin(np.array([a_space, b_space]), x_space)
    err_space = 0.5 * np.mean((y_space - ys) ** 2, axis=2)
    a_space, b_space = np.meshgrid(a_lin, b_lin)
    ax1.contourf(a_space, b_space, err_space, levels=81, cmap='magma')

    for app_idx in range(len(a_approx)):
        a, b = a_approx[app_idx], b_approx[app_idx]
        lab = '{}, a: {:.3f}, b: {:.3f}'.format(params[app_idx], a[-1], b[-1])
        ax1.plot(a, b, COLORS[app_idx], label=lab)
        ax1.plot(a[::len(a) // 10], b[::len(b) // 10], c=COLORS[app_idx], ls='', marker='*')
    ax1.plot(real[0], real[1], 'rx', label='real, a: {:.3f}, b: {:.3f}'.format(real[0], real[1]))
    ax1.set_xlabel('a'), ax1.set_ylabel('b')
    ax1.legend(fontsize='x-small')
    plt.show()


def extract_var(seq, idx):
    return [arr[idx] for arr in seq]


N, SIGMA, STEPS = 10000, 2.0, 500
a, b, xs, ys = generate_lin(N, SIGMA, x_range=(-10, 10), param_range=(-5, 5))
batches, args = [1, 10, 100, 1000, 10000], [f_lin, df_lin, 2]

approx = [Grad(*args)(xs, ys, batch_size, Grad.sqr_err, steps=STEPS) for batch_size in batches]
batches_str = ["Batch: {}".format(batch_size) for batch_size in batches]
plot_lin(xs, ys, approx, batches_str, (a, b))

initial, approx = np.random.uniform(-1, 1, size=2), []
opts = [Grad(*args), RMSProp(*args), AdaDelta(*args), Adam(*args)]
opts_str = ["vanilla", "rmsprop", "adadelta", "adam"]

for opt, name in zip(opts, opts_str):
    pr = profile.Profile()
    apx = pr.runcall(opt.__call__, xs, ys, 25, Grad.sqr_err, steps=STEPS, init=initial)
    print(name, end=': ')
    pr.print_stats()
    approx.append(apx)
plot_lin(xs, ys, approx, opts_str, (a, b))

for attempt in range(10):
    a, b, xs, ys = generate_lin(N, SIGMA, x_range=(-1000, 1000), param_range=(-10, 10))
    apx = Grad(*args)(xs, ys, 25, Grad.sqr_err, steps=STEPS)
    if math.isnan(apx[-1][0]) or math.isnan(apx[-1][1]):
        print(f'Attempt #{attempt}: Non-normalized does not converge')
    else:
        print(f'Attempt #{attempt}: Non-normalized converges')
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    xs = np.array([(x - x_min) / (x_max - x_min) for x in xs])
    ys = np.array([(y - y_min) / (y_max - y_min) for y in ys])
    apx = Grad(*args)(xs, ys, 25, Grad.sqr_err, steps=STEPS)
    if math.isnan(apx[-1][0]) or math.isnan(apx[-1][1]):
        print(f'Attempt #{attempt}: Normalized does not converge')
    else:
        print(f'Attempt #{attempt}: Normalized converges')
