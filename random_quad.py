"""Random quadratic function."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.stats import ortho_group
import pickle

from opt import OPT
import solvers_scaled as solvers1

from colors import *


def random_quad(d=50, eig_range=[1e-3, 1], 
              maxiter_tune=50, max_evals=500, 
              max_evals_rgd=1200, maxiter=100, me=10):
    
    M = ortho_group.rvs(dim=d)
    A = M.T.dot(np.diag(
            np.random.uniform(eig_range[0], eig_range[1], d)).dot(M))

    x0 = 5*np.ones(d)
    v0 = np.zeros(d)

    func = lambda x: 0.5*x.dot(A.dot(x))
    grad = lambda x: A.dot(x)
    
    def par_range(*names):
        d = {'stepsize': [1e-3, 1],
             'mu': [.5, .99],
             'delta': [0, 10],
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
   
    fig.savefig('figs/random_quad_rate.pdf', bbox_inches='tight')
    
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
    fig.savefig('figs/random_quad_hist.pdf', bbox_inches='tight')
  

if __name__ == '__main__':
    
    random_quad(d=500, eig_range=[1e-3, 10], 
              maxiter_tune=350, max_evals=500, 
              max_evals_rgd=1500, maxiter=700, me=50)
    
