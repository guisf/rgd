"""Matrix completion, nonconvex and alternating minimization approach.
We tune with Bayesian optimization.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, tpe, Trials, fmin

from skimage import color
from skimage import io

from colors import *

def make_support(shape, m):
    """Creates support for a matrix with uniform distribution.

    shape: (rows, columns)
        m: number of nonzero entries

    Output a boolean array of dimension ``shape'' with True 
    values on nonzero positions

    """
    total = shape[0]*shape[1]
    omega = np.zeros(total, dtype=int)
    ids = np.random.choice(range(total), m, replace=False)
    omega[ids] = 1
    omega = omega.reshape(shape)
    return omega

def obj(Mobs, Mask, A, B, mu):
    """Objective function."""
    return 0.5*np.linalg.norm(Mobs - Mask*A.dot(B.T)) + \
           0.5*mu*(np.linalg.norm(A) + np.linalg.norm(B))

def rel_error(Mtrue, Mhat):
    """Compute relative error with the ground truth."""
    return np.linalg.norm(Mtrue - Mhat)/np.linalg.norm(Mtrue)

def gradient_descent(Mtrue, Mobs, Mask, A, B, 
                    lamb=10, epsilon=0.5, maxiter=500):
    """Just for comparison."""
    h = epsilon 
    error = [rel_error(Mtrue, A.dot(B.T))]
    for k in range(maxiter):
        A = A + h*(Mask*(Mobs - A.dot(B.T))).dot(B) - h*lamb*A
        B = B + h*((Mask*(Mobs - A.dot(B.T))).T).dot(A) - h*lamb*B
        error.append(rel_error(Mtrue, A.dot(B.T)))
    return error

def tune_gradient_descent(Mtrue, Mobs, Mask, A, B, lamb, 
                          param_range, maxiter=100, max_evals=10):
    def tuning_obj(theta):
        epsilon = theta['epsilon']
        score = gradient_descent(Mtrue,Mobs,Mask,A,B,lamb,
                    epsilon=epsilon,maxiter=maxiter)
        return score[-1]
    space = {}
    space['epsilon'] = hp.uniform('epsilon', *param_range['epsilon'])
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_obj, space=space, algo=tpe_algo,
                        trials=tpe_trials, max_evals=max_evals)
    return tpe_trials, tpe_best

def classical_momentum(Mtrue, Mobs, Mask, A, B, Va, Vb, 
            lamb=10, epsilon=0.5, mu=0.9, maxiter=500):
    """Classical momentum or heavy ball method."""
    h = epsilon 
    error = [rel_error(Mtrue, A.dot(B.T))]
    for k in range(maxiter):
        Va = mu*Va - h*(-(Mask*(Mobs - A.dot(B.T))).dot(B) + lamb*A)
        A = A + Va
        Vb = mu*Vb - h*(-(Mask*(Mobs - A.dot(B.T))).T.dot(A) + lamb*B)
        B = B + Vb
        error.append(rel_error(Mtrue, A.dot(B.T)))
    return error

def tune_classical_momentum(Mtrue, Mobs, Mask, A, B, Va, Vb, lamb, 
            param_range, maxiter=100, max_evals=10):
    def tuning_obj(theta):
        epsilon = theta['epsilon']
        mu = theta['mu']
        score = classical_momentum(Mtrue,Mobs,Mask,A,B,Va,Vb,lamb,
                    epsilon=epsilon,mu=mu,maxiter=maxiter)
        return score[-1]
    space = {}
    space['epsilon'] = hp.uniform('epsilon', *param_range['epsilon'])
    space['mu'] = hp.uniform('mu', *param_range['mu'])
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_obj, space=space, algo=tpe_algo,
                        trials=tpe_trials, max_evals=max_evals)
    return tpe_trials, tpe_best

def nesterov(Mtrue, Mobs, Mask, A, B, Va, Vb, 
            lamb=10, epsilon=0.5, mu=0.9, maxiter=500):
    """Nesterov method."""
    h = epsilon 
    error = [rel_error(Mtrue, A.dot(B.T))]
    for k in range(maxiter):
        Ahat = A + mu*Va
        Va = mu*Va - h*(-(Mask*(Mobs - Ahat.dot(B.T))).dot(B) + lamb*Ahat)
        A = A + Va
        Bhat = B + mu*Vb
        Vb = mu*Vb - h*(-(Mask*(Mobs - A.dot(Bhat.T))).T.dot(A) + lamb*Bhat)
        B = B + Vb
        error.append(rel_error(Mtrue, A.dot(B.T)))
    return error

def tune_nesterov(Mtrue, Mobs, Mask, A, B, Va, Vb, lamb, 
            param_range, maxiter=100, max_evals=10):
    def tuning_obj(theta):
        epsilon = theta['epsilon']
        mu = theta['mu']
        score = nesterov(Mtrue,Mobs,Mask,A,B,Va,Vb,lamb,
                    epsilon=epsilon,mu=mu,maxiter=maxiter)
        return score[-1]
    space = {}
    space['epsilon'] = hp.uniform('epsilon', *param_range['epsilon'])
    space['mu'] = hp.uniform('mu', *param_range['mu'])
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_obj, space=space, algo=tpe_algo,
                        trials=tpe_trials, max_evals=max_evals)
    return tpe_trials, tpe_best

#def relativistic2(Mtrue, Mobs, Mask, A, B, Va, Vb, lamb, 
#            epsilon=0.5, mu=0.9, delta=0.5, alpha=1, maxiter=500):
#    """Our relativistic method that can interpolate between CM and NAG.
#    
#    set alpha=1, delta=0 for CM
#    set alpha=0, delta=0 for NAG
#    """
#    error = [rel_error(Mtrue, A.dot(B.T))]
#    for k in range(maxiter):
#        Ahalf = A + mu*Va/np.sqrt(delta*(mu**2)*np.linalg.norm(Va)**2+1)
#        Va = mu*Va-epsilon*(-(Mask*(Mobs-Ahalf.dot(B.T))).dot(B)+lamb*Ahalf)
#        A = alpha*Ahalf+(1-alpha)*A+Va/np.sqrt(delta*np.linalg.norm(Va)**2+1)
#        Bhalf = B + mu*Vb/np.sqrt(delta*(mu**2)*np.linalg.norm(Vb)**2+1)
#        Vb = mu*Vb-epsilon*(-(Mask*(Mobs-A.dot(Bhalf.T))).T.dot(A)+lamb*Bhalf)
#        B = alpha*Bhalf+(1-alpha)*B+Vb/np.sqrt(delta*np.linalg.norm(Vb)**2+1)
#        error.append(rel_error(Mtrue, A.dot(B.T)))
#    return error

def relativistic2(Mtrue, Mobs, Mask, A, B, Va, Vb, lamb, 
            epsilon=0.5, mu=0.9, delta=0.5, alpha=1, maxiter=500):
    """Our relativistic method that can interpolate between CM and NAG.
    
    set alpha=1, delta=0 for CM
    set alpha=0, delta=0 for NAG
    """
    error = [rel_error(Mtrue, A.dot(B.T))]
    for k in range(maxiter):
        Ahalf = A + np.sqrt(mu)*Va/np.sqrt(delta*mu*np.linalg.norm(Va)**2+1)
        Vahalf = np.sqrt(mu)*Va-epsilon*(
                    -(Mask*(Mobs-Ahalf.dot(B.T))).dot(B)+lamb*Ahalf)
        A = alpha*Ahalf+(1-alpha)*A+Vahalf/np.sqrt(
                                delta*np.linalg.norm(Vahalf)**2+1)
        Va = np.sqrt(mu)*Vahalf
        ##########################
        Bhalf = B + np.sqrt(mu)*Vb/np.sqrt(delta*mu*np.linalg.norm(Vb)**2+1)
        Vbhalf = np.sqrt(mu)*Vb-epsilon*(
                    -(Mask*(Mobs-A.dot(Bhalf.T))).T.dot(A)+lamb*Bhalf)
        B = alpha*Bhalf+(1-alpha)*B+Vbhalf/np.sqrt(
                                delta*np.linalg.norm(Vbhalf)**2+1)
        Vb = np.sqrt(mu)*Vbhalf

        error.append(rel_error(Mtrue, A.dot(B.T)))
    return error

def tune_relativistic(Mtrue, Mobs, Mask, A, B, Va, Vb, lamb, 
            param_range, maxiter=100, max_evals=10):
    def tuning_obj(theta):
        epsilon = theta['epsilon']
        mu = theta['mu']
        delta = theta['delta']
        alpha = theta['alpha']
        score = relativistic2(Mtrue,Mobs,Mask,A,B,Va,Vb,lamb,
              epsilon=epsilon,mu=mu,alpha=alpha,delta=delta,maxiter=maxiter)
        return score[-1]
    space = {}
    space['epsilon'] = hp.uniform('epsilon', *param_range['epsilon'])
    space['mu'] = hp.uniform('mu', *param_range['mu'])
    space['delta'] = hp.uniform('delta', *param_range['delta'])
    space['alpha'] = hp.uniform('alpha', *param_range['alpha'])
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=tuning_obj, space=space, algo=tpe_algo,
                        trials=tpe_trials, max_evals=max_evals)
    return tpe_trials, tpe_best

def matrix_completion(n1=100, n2=100, r=2, sr=0.15, 
        tune_iter=100, tune_evals=200, tune_evals_rgd=1000, maxiter=150, me=10):
    #n1,n2 = (100,100) # matrix size
    #r = 2  # rank
    #sr = 0.15 # sampling ratio
    p = int(sr*n1*n2) # number sampled entries
    print("-> Number of observed entries:", p)
    print("-> |Omega| ~", n1*r)
    d = r*(n1+n2-r) # effective degrees of freedom per measurement
    print("-> Hardness:", d/p)
    lamb = 0

    # Generate data
    Ml = np.random.normal(1,2,size=(n1,r))
    Mr = np.random.normal(1,2,size=(n2,r))
    M = Ml.dot(Mr.T)
    Om = make_support(M.shape, p) # support of observed entries
    Mobs = Om*M

    # Initialization
    rhat = r
    A = np.random.normal(0, 1, (n1, rhat))
    B = np.random.normal(0, 1, (n2, rhat))
    #A = np.ones((n1,rhat))
    #B = np.ones((n2,rhat))
    Va = np.zeros((n1, rhat))
    Vb = np.zeros((n2, rhat))

    # GD for comparison
    gd_trials, gd_best = tune_gradient_descent(M, Mobs, Om, A, B, lamb, 
            {'epsilon':[1e-5, 1e-2]}, maxiter=tune_iter,max_evals=tune_evals)
    print('\n*** GD params:', gd_best)
    gd = gradient_descent(M,Mobs,Om,A,B,lamb,
            gd_best['epsilon'],maxiter=maxiter)
    
    # NAG
    nag_trials, nag_best = tune_nesterov(M, Mobs, Om, A, B, Va, Vb, lamb, 
            {'epsilon': [1e-5, 1e-2], 'mu': [0.3, 0.99]}, 
             maxiter=tune_iter, max_evals=tune_evals)
    print('\n*** NAG params:', nag_best, '\n')
    nag =  nesterov(M, Mobs, Om, A, B, Va, Vb, lamb, 
                maxiter=maxiter, **nag_best) 
    # CM 
    cm_trials, cm_best = tune_classical_momentum(M, Mobs, Om, A, B, Va, 
        Vb, lamb, {'epsilon': [1e-5, 1e-2], 'mu': [0.3, 0.99]}, 
        maxiter=tune_iter,max_evals=tune_evals)
    print('\n*** CM params:', cm_best, '\n')
    cm =  classical_momentum(M, Mobs, Om, A, B, Va, Vb, lamb, maxiter=maxiter, 
                                **cm_best) 
    # RGD
    rgd_trials, rgd_best = tune_relativistic(M, Mobs, Om, A, B, Va, Vb, lamb, 
        {'epsilon': [1e-5, 1e-2],'mu':[0.3,0.99],'alpha':[0,1],'delta':[0,10]},
        maxiter=tune_iter, max_evals=tune_evals_rgd)
    print('\n*** RGD params:', rgd_best, '\n')
    rgd =  relativistic2(M, Mobs, Om, A, B, Va, Vb, lamb, 
                        maxiter=maxiter, **rgd_best)
   
    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    ax.plot(gd, label='GD', color=color_gd, marker=marker_gd, linestyle=linestyle_gd, markevery=me)
    ax.plot(cm, label='CM', color=color_cm, marker=marker_cm, linestyle=linestyle_cm, markevery=me)
    ax.plot(nag, label='NAG', color=color_nag, marker=marker_nag, linestyle=linestyle_nag, markevery=me)
    ax.plot(rgd, label='RGD', color=color_rgd, marker=marker_rgd, linestyle=linestyle_rgd, markevery=me)
    ax.legend()
    ax.set_ylabel(r'$\| M - \hat{M}_k\|/\| M\|$')
    ax.set_xlabel(r'$k$')
    fig.savefig('mat_comp_rate.pdf', bbox_inches='tight')

    fig, axs = plt.subplots(2, 2)
    alpha = .5
    axs[0,0].hist(gd_trials.idxs_vals[1]['epsilon'], alpha=alpha,
                density=True, color=color_gd)
    axs[0,0].hist(cm_trials.idxs_vals[1]['epsilon'], alpha=alpha,
                density=True, color=color_cm)
    axs[0,0].hist(nag_trials.idxs_vals[1]['epsilon'], alpha=alpha,
                density=True, color=color_nag)
    axs[0,0].hist(rgd_trials.idxs_vals[1]['epsilon'], alpha=alpha,
                density=True, color=color_rgd)
    axs[0,0].set_title(r'$\epsilon$')

    axs[0,1].hist(cm_trials.idxs_vals[1]['mu'], alpha=alpha,
                density=True, color=color_cm)
    axs[0,1].hist(nag_trials.idxs_vals[1]['mu'], alpha=alpha,
                density=True, color=color_nag)
    axs[0,1].hist(rgd_trials.idxs_vals[1]['mu'], alpha=alpha,
                density=True, color=color_rgd)
    axs[0,1].set_title(r'$\mu$')

    axs[1,0].hist(rgd_trials.idxs_vals[1]['delta'], alpha=alpha,
                density=True, color=color_rgd)
    axs[1,0].set_title(r'$\delta$')

    axs[1,1].hist(rgd_trials.idxs_vals[1]['alpha'], alpha=alpha,
                density=True, color=color_rgd)
    axs[1,1].set_title(r'$\alpha$')

    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    fig.savefig('mat_comp_hist.pdf', bbox_inches='tight')


if __name__ == '__main__':
    matrix_completion(n1=100, n2=100, r=5, sr=0.3, 
            tune_iter=200, tune_evals=600, tune_evals_rgd=1500, maxiter=200, me=15)

