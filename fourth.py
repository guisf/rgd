"""Correlated quadratic function."""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from opt import OPT
import solvers_scaled as solvers1


def fourth_power(d=1, maxiter_tune=50, max_evals=500, 
              max_evals_rgd=1200, maxiter=100):
    
    func = lambda x: (1./8.)*(x[0]**2 + 1)**4 - 1./8.
    grad = lambda x: np.array([x[0]*(x[0]**2+1)**3])
    
    x0 = 1.5*np.ones(d)
    v0 = np.zeros(d)
    
    def par_range(*names):
        d = {'stepsize': [1e-4, 9e-1],
             'mu': [.3, .9],
             'delta': [0, 100],
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
    #rgd.tune(par_range('stepsize', 'mu', 'delta'), 
             maxiter=maxiter_tune, max_evals=max_evals_rgd)
    rgd.solve(maxiter=maxiter)
    
    # relativistic 2
    #rgd2 = OPT(func, grad, x0, v0)
    #rgd2.set_solver(solvers1.relativistic3)
    #rgd.tune(par_range('stepsize', 'mu', 'delta', 'alpha'), 
    #rgd2.tune(par_range('stepsize', 'mu', 'delta'), 
    #         maxiter=maxiter_tune, max_evals=max_evals_rgd)
    #rgd2.solve(maxiter=maxiter)

    print('cm', cm.best)
    print('nag', nag.best)
    print('rgd', rgd.best)
    #print('rgd2', rgd2.best)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(cm.scores, label='CM', color='mediumblue')
    ax.plot(nag.scores, label='NAG', color='olivedrab')
    ax.plot(rgd.scores, label='RGD', color='black')
    #ax.plot(rgd2.scores, label='RGD2', color='red')
    ax.legend()
    #ax.set_ylabel(r'$f(x)$')
    #ax.set_xlabel(r'$k$')
    ax.set_xlabel(r'iteration')
   
    fig.savefig('figs/fourth_rate.pdf', bbox_inches='tight')
    
    fig, axs = plt.subplots(2, 2)
    alpha = .5
    axs[0,0].hist(cm.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color='mediumblue')
    axs[0,0].hist(nag.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color='olivedrab')
    axs[0,0].hist(rgd.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color='black')
    #axs[0,0].hist(rgd2.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
    #            density=True, color='red')
    axs[0,0].set_title(r'$\epsilon$')
    
    axs[0,1].hist(cm.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color='mediumblue')
    axs[0,1].hist(nag.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color='olivedrab')
    axs[0,1].hist(rgd.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color='black')
    #axs[0,1].hist(rgd2.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
    #            density=True, color='red')
    axs[0,1].set_title(r'$\mu$')

    axs[1,0].hist(rgd.tpe_trials.idxs_vals[1]['delta'], alpha=alpha, 
                density=True, color='black')
    #axs[1,0].hist(rgd2.tpe_trials.idxs_vals[1]['delta'], alpha=alpha, 
    #            density=True, color='red')
    axs[1,0].set_title(r'$\delta$')
    
    axs[1,1].hist(rgd.tpe_trials.idxs_vals[1]['alpha'], alpha=alpha, 
                density=True, color='black')
    axs[1,1].set_title(r'$\alpha$')
    
    plt.subplots_adjust(hspace=0.45, wspace=0.2)
    fig.savefig('figs/fourth_hist.pdf', bbox_inches='tight')
  

if __name__ == '__main__':
    
    fourth_power(d=1, maxiter_tune=100, max_evals=200, 
              max_evals_rgd=400, maxiter=100)
    
    #quad_corr(d=50, rho=0.7, maxiter_tune=40, max_evals=500, 
    #          max_evals_rgd=1000, maxiter=60)

