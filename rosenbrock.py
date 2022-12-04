"""Rosenbrock function."""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from opt import OPT
import solvers_scaled as solvers1

from colors import *

def rosenbrock(x0, v0, maxiter_tune=50, max_evals=500, 
              max_evals_rgd=1200, maxiter=100, me=10):

    def func(x):
        """Rosenbrock function. Vectorial form is much faster."""
        xp = x[1:]
        xm = x[:-1]
        return (100*(xp-xm**2)**2 + (1-xm)**2).sum()

    def grad(x):
        """Gradient of the above function in vectorized form."""
        G = np.zeros(len(x))
        G[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - \
                    2*(1-x[1:-1]) + 200*(x[1:-1]-x[:-2]**2)
        G[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
        G[-1] = 200*(x[-1]-x[-2]**2)
        return G
    
    def par_range(*names):
        d = {'stepsize': [1e-5, 1e-2],
             'mu': [.9, .99],
             'delta': [0, 30],
             'alpha': [0, 1]
        }
        return {n: d[n] for n in names}
    
    # classical momentum 
    cm = OPT(func, grad, x0, v0)
    cm.set_solver(solvers1.classical_momentum)
    cm.tune(par_range('stepsize', 'mu'), maxiter=maxiter_tune, 
                        max_evals=max_evals)
    cm.solve(maxiter=maxiter)
    
    # nesterov
    nag = OPT(func, grad, x0, v0)
    nag.set_solver(solvers1.nesterov)
    nag.tune(par_range('stepsize', 'mu'), maxiter=maxiter_tune, 
             max_evals=max_evals)
    nag.solve(maxiter=maxiter)
    
    # relativistic
    rgd = OPT(func, grad, x0, v0)
    rgd.set_solver(solvers1.relativistic2)
    rgd.tune(par_range('stepsize', 'mu', 'delta', 'alpha'), 
             maxiter=maxiter_tune, max_evals=max_evals_rgd)
    rgd.solve(maxiter=maxiter)
    
    print('cm', cm.best)
    print('nag', nag.best)
    print('rgd', rgd.best)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(cm.scores,label='CM',color=color_cm,marker=marker_cm,linestyle=linestyle_cm,markevery=me)
    ax.plot(nag.scores,label='NAG',color=color_nag,marker=marker_nag,linestyle=linestyle_nag,markevery=me)
    ax.plot(rgd.scores,label='RGD',color=color_rgd,marker=marker_rgd,linestyle=linestyle_rgd,markevery=me)
    ax.legend()
    ax.set_ylabel(r'$f(x)$')
    ax.set_xlabel(r'$k$')
    fig.savefig('figs/ros_rate.pdf', bbox_inches='tight')
    
    fig, axs = plt.subplots(2, 2)
    alpha = .5
    axs[0,0].hist(cm.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_cm)
    axs[0,0].hist(nag.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_nag)
    axs[0,0].hist(rgd.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_rgd)
    axs[0,0].set_title(r'$\epsilon$')
    
    axs[0,1].hist(cm.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_cm)
    axs[0,1].hist(nag.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_nag)
    axs[0,1].hist(rgd.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_rgd)
    axs[0,1].set_title(r'$\mu$')

    axs[1,0].hist(rgd.tpe_trials.idxs_vals[1]['delta'], alpha=alpha, 
                density=True, color=color_rgd)
    axs[1,0].set_title(r'$\delta$')
    
    axs[1,1].hist(rgd.tpe_trials.idxs_vals[1]['alpha'], alpha=alpha, 
                density=True, color=color_rgd)
    axs[1,1].set_title(r'$\alpha$')
    
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    fig.savefig('figs/ros_hist.pdf', bbox_inches='tight')
  

if __name__ == '__main__':
    
    x0 = 2*np.ones(100)
    for i in range(0, len(x0), 2):
        x0[i] = -2
    v0 = np.zeros(100)
    
    rosenbrock(x0, v0, maxiter_tune=1000, max_evals=500, 
               max_evals_rgd=1000, maxiter=2000, me=150)

