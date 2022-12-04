"""Example using relativistic methods and with Bayesian optimization 
for parameteri tuning. 

Guilherme Franca <guifranca@gmail.com>
June 2020
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# this is for Bayesian optimization
from hyperopt import hp, tpe, Trials, fmin
from hyperopt.pyll import stochastic


### Some basic optimization solvers ###

def heavy_ball(x0, v0, function, gradient, maxiter=1000,
               stepsize=1e-3, mu=0.9):
    """heavy ball method."""

    xs = [x0]
    x = x0
    v = v0
    for k in range(maxiter):

        g = gradient(x)
        v = mu*v - stepsize*g
        x = x + v

        xs.append(x)

    fs = [function(x) for x in xs]
    return fs, xs

def nesterov(x0, v0, function, gradient, maxiter=1000, stepsize=1e-3, mu=0.9):
    """Nesterov's accelerated gradient."""
    
    xs = [x0]
    x = x0
    v = v0
    for k in range(maxiter):

        y = x + mu*v
        g = gradient(y)
        v = mu*v - stepsize*g
        x = x + v

        xs.append(x)

    fs = [function(x) for x in xs]
    return fs, xs

def relativistic(x0, v0, function, gradient, maxiter=1000,
                 stepsize=1e-3, mu=0.9, eta=0.1):
    """Relativistic method for optimization."""
    
    xs = [x0]
    x = x0
    v = v0
    g = gradient(x)
    for k in range(maxiter):

        v = v - stepsize*g
        x = x + v/np.sqrt(1+eta*v.dot(v))
        v = mu*v 
        #v = v + (1.-mu)*(1.-1./np.sqrt(1+delta*v.dot(v)))*v
        x = x + v/np.sqrt(1+eta*v.dot(v))
        g = gradient(x)
        v = v - stepsize*g

        xs.append(x)

    fs = [function(x) for x in xs]
    return fs, xs


### Example ###
if __name__ == '__main__':
    
    from scipy.stats import ortho_group

    # objective function and its gradient; random quadratic
    d = 100
    M = ortho_group.rvs(dim=d)
    A = M.T.dot(np.diag(np.random.uniform(0.0, 1, d)).dot(M))
    func = lambda x: 0.5*x.dot(A.dot(x))
    grad = lambda x: A.dot(x)
    
    # initial state
    x0 = 1*np.ones(d)
    v0 = np.zeros(d)
    maxiter_tune = 200
    maxiter = 500

    # define objective function to tune the algorithm
    # this function can only accepts a dictionary of parameters and return
    # a score number
    def tuning_objective_nesterov(params):
        stepsize, mu = params['stepsize'], params['mu']
        scores, _ = nesterov(x0,v0,func,grad,maxiter_tune,stepsize,mu)
        return scores[-1]
    
    def tuning_objective_heavy_ball(params):
        stepsize, mu = params['stepsize'], params['mu']
        scores, _ = heavy_ball(x0,v0,func,grad,maxiter_tune,stepsize,mu)
        return scores[-1]
    
    def tuning_objective_relativistic(params):
        stepsize, mu, eta = params['stepsize'], params['mu'], params['eta']
        scores, _ = relativistic(x0,v0,func,grad,maxiter_tune,stepsize,mu,eta)
        return scores[-1]

    # space for parameter search
    space = {
        'stepsize': hp.uniform('stepsize', 1e-6, 1),
        'mu': hp.uniform('mu', 0.5, 0.99),
        'eta': hp.uniform('eta', 0, 50)
    }
    
    # Bayesian optimization tuning 
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_objective_nesterov,space=space,algo=tpe_algo,
                    trials=tpe_trials,max_evals=200)
    par_nest = tpe_best
    del par_nest['eta']
    
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_objective_heavy_ball,space=space,algo=tpe_algo,
                    trials=tpe_trials,max_evals=200)
    par_heavy= tpe_best
    del par_heavy['eta']
    
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_objective_relativistic,space=space,algo=tpe_algo,
                    trials=tpe_trials,max_evals=500)
    par_rel = tpe_best

    print('Params Nesterov: ', par_nest)
    print('Params heavy ball: ', par_heavy)
    print('Params relativistic: ', par_rel)

    ### Let's run the actual algorithms now with the best parameters ###
    nest_scores, _ = nesterov(x0,v0,func,grad,maxiter=maxiter,
                        stepsize=par_nest['stepsize'],mu=par_nest['mu'])
    heavy_scores, _ = heavy_ball(x0,v0,func,grad,maxiter=maxiter,
                        stepsize=par_heavy['stepsize'],mu=par_heavy['mu'])
    rel_scores, _ = relativistic(x0,v0,func,grad,maxiter=maxiter,
            stepsize=par_rel['stepsize'],mu=par_rel['mu'],eta=par_rel['eta'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(nest_scores, label='nesterov')
    ax.plot(heavy_scores, label='heavy ball')
    ax.plot(rel_scores, label='relativistic')
    ax.legend()
    fig.savefig('./test.pdf')

