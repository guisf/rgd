# Example of tuning the optimization algorithms using Bayesian optimization 
#
# Guilherme S. Franca <guifranca@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, tpe, Trials, fmin

from opt_solvers import *

# we use a simple quadratic function
d = 50
r = int(0.8*d)
A = np.random.normal(loc=0, scale=1, size=(r,d))
M = A.T.dot(A)/d
x0 = 10*np.ones(d)

func = lambda x: x.dot(M.dot(x))/2.
grad = lambda x: M.dot(x)

def tune(param_range, algo, maxiter=1000, max_evals=100):
    """Function used to tune the optimization solver."""
        
    def tune_func(theta):
        xs = algo(x0, grad, maxiter=maxiter, **theta)
        x = xs[-1]
        return func(x)

    space = {}  # range for parameter search
    for par_name, par_range in param_range.items():
        space[par_name] = hp.uniform(par_name, *par_range)

    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tune_func, space=space, algo=tpe_algo, 
                        trials=tpe_trials, max_evals=max_evals)
    return tpe_best


if __name__ == "__main__":

    max_iter_tune = 150 # number of iterations to tune
    max_evals = 400 # number of evaluations to tune
    max_iter = 300 # number of iterations for runing with best parameters

    # tuning the algorithms
    cm_par = tune({'stepsize': [1e-4,1e-1], 'mu': [0.5,0.99]},
                  classical_momentum, 
                  maxiter=max_iter_tune,  max_evals=max_evals)
    nag_par = tune({'stepsize': [1e-4,1e-1], 'mu': [0.5,0.99]},
                   nesterov,
                   maxiter=max_iter_tune,  max_evals=max_evals)
    rgd_par = tune({'stepsize':[1e-4,1e-1], 'mu':[0.5,0.99], 'delta':[0,10]},
                   relativistic, 
                   maxiter=max_iter_tune,  max_evals=max_evals)
    print()
    print("Best parameters")
    print("CM:", cm_par)
    print("NAG:", nag_par)
    print("RGD:", rgd_par)
    print()

    # runing the algorithms with the best parameters found
    xs = classical_momentum(x0, grad, maxiter=max_iter, **cm_par)
    fs_cm = [func(x) for x  in xs]
    xs = nesterov(x0, grad, maxiter=max_iter, **nag_par)
    fs_nag = [func(x) for x  in xs]
    xs = relativistic(x0, grad, maxiter=max_iter, **rgd_par)
    fs_rgd = [func(x) for x  in xs]
    
    # ploting the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(fs_cm, label='CM')
    ax.plot(fs_nag, label='NAG')
    ax.plot(fs_rgd, label='RGD')
    ax.legend(loc=0)
    fig.savefig('test.pdf')

