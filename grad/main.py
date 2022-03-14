import pandas as pd
from grad import Gladiator
from schedulers import *
from quadra import Quadratic


def sin_cos(v):
    sin_arg = 0.5 * v[0] ** 2 - 0.25 * v[1] ** 2 + 3
    cos_arg = 2 * v[0] + 1 - np.exp(v[1])
    return np.sin(sin_arg) * np.cos(cos_arg)


def d_sin_cos(v):
    sin_arg = 0.5 * v[0] ** 2 - 0.25 * v[1] ** 2 + 3
    cos_arg = 2 * v[0] + 1 - np.exp(v[1])
    dfx = v[0] * np.cos(sin_arg) * np.cos(cos_arg)
    dfx -= 2 * np.sin(sin_arg) * np.sin(cos_arg)
    dfy = -0.5 * v[1] * np.cos(sin_arg) * np.cos(cos_arg)
    dfy += np.sin(sin_arg) * np.sin(cos_arg) * np.exp(v[1])
    return np.array([dfx, dfy])


def simple_quad(v):
    return 10 * v[0] ** 2 + v[1] ** 2


def d_simple_quad(v):
    return np.array([20 * v[0], 2 * v[1]])


def quad(v):
    return v[0] ** 2 + 2 * v[0] * v[1] + 4 * v[1] ** 2 - 5 * v[0] - 3 * v[1] + 1


def d_quad(v):
    return np.array([2 * v[0] + 2 * v[1] - 5, 2 * v[0] + 8 * v[1] - 3])


def rosenbrock(v):
    return (1 - v[0]) ** 2 + 5 * (v[1] - v[0] ** 2) ** 2


def d_rosenbrock(v):
    dfx = 20 * v[0] ** 3 - 20 * v[0] * v[1] + 2 * v[0] - 2
    dfy = 10 * (v[1] - v[0] ** 2)
    return np.array([dfx, dfy])


F = [(sin_cos, d_sin_cos), (simple_quad, d_simple_quad), (quad, d_quad), (rosenbrock, d_rosenbrock)]
FN, RES_DIR = list(map(lambda fun: fun[0].__name__, F)), 'results/'
RUNS, STEPS, EPS, best_runs = 10, 5000, 1e-3, {}


print("- task 1. Constant learn rate")
LR = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
stats, best = np.empty(shape=(len(LR), len(F))), {}

for f_idx, (f, df) in enumerate(F):
    best_x, best_f, best_lr = None, [float('inf')], None
    for lr_idx, lr in enumerate(LR):
        scheduler = ConstScheduler(lr)
        f_eval, gladiator = 0, Gladiator(f, 2, df, scheduler)
        for run in range(RUNS):
            try:
                x_values, f_values = gladiator(STEPS, EPS)
                if f_values[-1] < best_f[-1]:
                    best_x, best_f, best_lr = x_values, f_values, lr
                f_eval += len(f_values)
            except RuntimeWarning:
                f_eval += STEPS
        stats[lr_idx, f_idx] = f_eval / RUNS
    best[f] = best_x, best_f, "lr={}".format(best_lr)
best_runs['const_lr'] = best
pd.DataFrame(stats, columns=FN).to_csv(RES_DIR + 'const_lr/st')


print("- task 2. Dynamic learn rate")
EXP_K, LR = [1e-6, 1e-5, 1e-4, 1e-3], 1e-2
stats, best = np.empty(shape=(len(EXP_K), len(F))), {}

for f_idx, (f, df) in enumerate(F):
    best_x, best_f, best_k = None, [float('inf')], None
    for k_idx, k in enumerate(EXP_K):
        scheduler = ExpScheduler(LR, k)
        f_eval, gladiator = 0, Gladiator(f, 2, df, scheduler)
        for run in range(RUNS):
            try:
                x_values, f_values = gladiator(STEPS, EPS)
                if f_values[-1] < best_f[-1]:
                    best_x, best_f, best_k = x_values, f_values, k
                f_eval += len(f_values)
            except RuntimeWarning:
                f_eval += STEPS
        stats[k_idx, f_idx] = f_eval / RUNS
    best[f] = best_x, best_f, "k={}".format(best_k)
best_runs['dynamic_lr'] = best
pd.DataFrame(stats, columns=FN) .to_csv(RES_DIR + 'dynamic_lr/st')


print("- task 3-4. 1D-search")
ALPHA_SPACE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
stats, best = np.empty(shape=len(F)), {}

for f_idx, (f, df) in enumerate(F):
    scheduler = ArmijoScheduler(f, df, ALPHA_SPACE)
    f_eval, gladiator = 0, Gladiator(f, 2, df, scheduler)
    best_x, best_f = [], [float('inf')]
    for run in range(RUNS):
        try:
            x_values, f_values = gladiator(STEPS, EPS)
            if f_values[-1] < best_f[-1]:
                best_x, best_f = x_values, f_values
            f_eval += len(f_values) * (1 + len(ALPHA_SPACE))
        except RuntimeWarning:
            f_eval += STEPS * (1 + len(ALPHA_SPACE))
    stats[f_idx] = f_eval / RUNS
    best[f] = best_x, best_f, ""
best_runs['wolfe'] = best
pd.Series(stats, index=FN).to_csv(RES_DIR + "wolfe/st")


print("- task 5. Trajectory")
for task, trajectories in best_runs.items():
    for fun, (x_values, f_values, param) in trajectories.items():
        description = "{}, {}".format(fun.__name__, param)
        Gladiator.plot_two_dim(x_values, f_values, fun, description)


print("- task 6. Quadratic problem")
N_INIT, K_INIT = 100, 10
N_MAX, K_MAX = 1600, 160
N_GRID, K_GRID = 15, 15

ns = np.linspace(N_INIT, N_MAX, N_GRID, dtype=int)
ks = np.linspace(K_INIT, K_MAX, K_GRID, dtype=int)
stats = np.zeros(shape=(N_GRID, K_GRID))

for n_idx, n in enumerate(ns):
    for k_idx, k in enumerate(ks):
        for run in range(RUNS):
            q = Quadratic(n, k)
            scheduler = ArmijoScheduler(q.f, q.df, ALPHA_SPACE)
            gladiator = Gladiator(q.f, n, q.df, scheduler)
            xs, fs = gladiator(STEPS, EPS)
            stats[n_idx, k_idx] += len(fs) / RUNS
Quadratic.plot_quads(ns, ks, stats)
