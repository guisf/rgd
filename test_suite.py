"""
Test suite for different objective functions.

See https://arxiv.org/pdf/1308.4008.pdf for details

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import pickle

from opt import OPT
import solvers_scaled as solvers1

from colors import *


def test(x0,v0,func,grad,par_search,maxiter_tune=50,max_evals=500,
         max_evals_rgd=1200, maxiter=100, markevery=10, fname='plot'):
    print('**** Doing: %s ****'%func.__name__)
    
    def par_range(*names):
        d = par_search # dictionary with par name and range
        return {n: d[n] for n in names}

    ##########################################################################
    # classical momentum
    cm = OPT(func, grad, x0, v0)
    cm.set_solver(solvers1.classical_momentum)
    cm.tune(par_range('stepsize','mu'),maxiter=maxiter_tune,
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
    print('** best tuning parameters **')
    print('cm', cm.best)
    print('nag', nag.best)
    print('rgd', rgd.best)

    ##########################################################################
    # plotting
    fig = plt.figure(constrained_layout=True)
    w, h = fig.get_size_inches()
    fig.set_size_inches(2*w, h)
    gs = fig.add_gridspec(nrows=2, ncols=4)
    ax = fig.add_subplot(gs[0:2, 0:2]) # convergence rate
    ax.set_yscale('log')
    ax.plot(cm.scores, label='CM',color=color_cm,marker=marker_cm,
            linestyle=linestyle_cm,markevery=markevery)
    ax.plot(nag.scores, label='NAG',color=color_nag,marker=marker_nag,
            linestyle=linestyle_nag,markevery=markevery)
    ax.plot(rgd.scores, label='RGD',color=color_rgd,marker=marker_rgd,
            linestyle=linestyle_rgd,markevery=markevery)
    ax.legend()
    ax.set_ylabel(r'$f(x)-f^\star$')
    ax.set_xlabel(r'$k$')
    # histograms
    axs_00 = fig.add_subplot(gs[0, 2]) # histogram subplots
    axs_01 = fig.add_subplot(gs[0, 3])
    axs_10 = fig.add_subplot(gs[1, 2])
    axs_11 = fig.add_subplot(gs[1, 3])
    alpha = .5
    axs_00.hist(cm.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_cm)
    axs_00.hist(nag.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_nag)
    axs_00.hist(rgd.tpe_trials.idxs_vals[1]['stepsize'], alpha=alpha, 
                density=True, color=color_rgd)
    axs_00.set_title(r'$\epsilon$')
    axs_01.hist(cm.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_cm)
    axs_01.hist(nag.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_nag)
    axs_01.hist(rgd.tpe_trials.idxs_vals[1]['mu'], alpha=alpha, 
                density=True, color=color_rgd)
    axs_01.set_title(r'$\mu$')
    axs_10.hist(rgd.tpe_trials.idxs_vals[1]['delta'], alpha=alpha, 
                density=True, color=color_rgd)
    axs_10.set_title(r'$\delta$')
    axs_11.hist(rgd.tpe_trials.idxs_vals[1]['alpha'], alpha=alpha, 
                density=True, color=color_rgd)
    axs_11.set_title(r'$\alpha$')
    
    #fig.tight_layout()
    fig.savefig('figs/%s.pdf'%fname, bbox_inches='tight')


##############################################################################
# several test functions below

def ros(x):
    xp = x[1:]
    xm = x[:-1]
    return (100*(xp-xm**2)**2 + (1-xm)**2).sum()

def ros_grad(x):
    G = np.zeros(len(x))
    G[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - \
                    2*(1-x[1:-1]) + 200*(x[1:-1]-x[:-2]**2)
    G[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
    G[-1] = 200*(x[-1]-x[-2]**2)
    return G

def beale(x):
    a = x[0]
    b = x[1]
    return (1.5-a+a*b)**2 + (2.25-a+a*(b**2))**2+(2.625-a+a*(b**3))**2

def beale_grad(x):
    a = x[0]
    b = x[1]
    xcomp = 2*(1.5-a+a*b)*(-1+b) + \
            2*(2.25-a+a*(b**2))*(-1+(b**2)) + \
            2*(2.625-a+a*(b**3))*(-1+(b**3))
    ycomp = 2*(1.5-a+a*b)*(a) + \
            2*(2.25-a+a*(b**2))*(2*a*b) + \
            2*(2.625-a+a*(b**3))*(3*a*(b**2))
    return np.array([xcomp, ycomp])

def rastrigin(x):
    A = 10
    return A*len(x) + np.sum(x**2-A*np.cos(2*np.pi*x))

def rastrigin_grad(x):
    A = 10
    return 2*x + A*2*np.pi*np.sin(2*np.pi*x)

def ackley(x):
    return -200*np.exp(-0.02*np.linalg.norm(x)) + 200

def ackley_grad(x):
    return -200*np.exp(-0.02*np.linalg.norm(x))*(-0.02*x/np.linalg.norm(x))

def ackley3(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    n = len(x)
    return -a*np.exp(-b*np.linalg.norm(x)/np.sqrt(n)) - \
            np.exp(np.sum(np.cos(c*x))/n) + a + np.exp(1)

def ackley3_grad(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    n = len(x)
    return a*b*np.exp(-b*np.linalg.norm(x)/np.sqrt(n))*x/(np.sqrt(n)*np.linalg.norm(x)) + np.exp(np.sum(np.cos(c*x)))*c*np.sin(c*x)/n
            
def goldstein(z):
    x = z[0]
    y = z[1]
    return (1 + (1 + x + y)**2*(19 - 14*x + 3*x**2 - 14*y + 6*x*y +  \
      3*y**2))*(30 + (2*x - 3*y)**2*(18 - 32*x + 12*x**2 + 48*y - \
      36*x*y + 27*y**2))

def goldstein_grad(z):
    x = z[0]
    y = z[1]
    gx = 24*(-1 + 2*x - 3*y)*(2*x - 3*y)*(2*x - \
    3*(1 + y))*(1 + (1 + x + y)**2*(19 + 3*x**2 + y*(-14 + 3*y) + \
       2*x*(-7 + 3*y))) + 12*(-2 + x + y)*(-1 + x + y)*(1 + x + \
        y)*(30 + (2*x - 3*y)**2*(18 + 12*x**2 - 4*x*(8 + 9*y) + \
        3*y*(16 + 9*y)))
    gy = -36*(-1 + 2*x - 3*y)*(2*x - 3*y)*(2*x - \
    3*(1 + y))*(1 + (1 + x + y)**2*(19 + 3*x**2 + y*(-14 + 3*y) + \
       2*x*(-7 + 3*y))) + \
 12*(-2 + x + y)*(-1 + x + y)*(1 + x + \
    y)*(30 + (2*x - 3*y)**2*(18 + 12*x**2 - 4*x*(8 + 9*y) + \
       3*y*(16 + 9*y)))
    return np.array([gx,gy])

def booth(z):
    x = z[0]
    y = z[1]
    return (x+2*y-7)**2 + (2*x+y-5)**2 

def booth_grad(z):
    x = z[0]
    y = z[1]
    return np.array([2*(x+2*y-7)*(1) + 4*(2*x+y-5), 
                     2*(x+2*y-7)*(2) + 2*(2*x+y-5)])

def bukin(z):
    x, y = z
    return 100*np.sqrt(np.abs(y-0.01*x**2)) + 0.01*np.abs(x+10)

def bukin_grad(z):
    x, y = z
    gx = 50*np.sign(y-0.01*x**2)*(-0.02*x)/np.sqrt(np.abs(y-0.01*x**2)) + \
            0.01*np.sign(x+10)
    gy = 50*np.sign(y-0.01*x**2)/np.sqrt(np.abs(y-0.01*x**2))
    return np.array([gx,gy])

def matyas(z):
    x, y = z
    return 0.26*(x**2+y**2)-0.48*x*y

def matyas_grad(z):
    x, y = z
    return np.array([0.26*2*x-0.48*y, 0.26*2*y-0.48*x]) 

def levi(z):
    x, y = z
    return (np.sin(3*np.pi*x))**2 + (x-1)**2*(1+(np.sin(3*np.pi*y))**2) \
            + (y-1)**2*(1+(np.sin(2*np.pi*y))**2)

def levi_grad(z):
    x, y = z
    return np.array([
            2*np.sin(3*np.pi*x)*np.cos(3*np.pi*x)*3*np.pi + \
            2*(x-1)*(1+(np.sin(3*np.pi*y))**2), 
            (x-1)**2*(2*np.sin(3*np.pi*y)*np.cos(3*np.pi*y)*3*np.pi) + \
            2*(y-1)*(1+(np.sin(2*np.pi*y))**2) + \
            (y-1)**2*(2*np.sin(2*np.pi*y)*np.cos(2*np.pi*y)*2*np.pi)])

def camel(z):
    x, y = z
    return 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2

def camel_grad(z):
    x, y = z
    g1 = 4*x - 1.05*4*x**3 + x**5 + y
    g2 = x + 2*y
    return np.array([g1, g2])

def styblinski(x):
    return np.sum(x**4 - 16*x**2 + 5*x)/2 + 39.16599*len(x)

def styblinski_grad(x):
    return (4*x**3 - 32*x + 5.)/2.

def brent(x):
    return (x[0]+10)**2 + (x[1]+10)**2 + np.exp(-np.linalg.norm(x)**2)

def brent_grad(x):
    return 2*(x+10) + np.exp(-np.linalg.norm(x)**2)*(-2*x)

def chung_reynolds(x):
    return np.linalg.norm(x)**4

def chung_reynolds_grad(x):
    return 4*np.linalg.norm(x)**2*(x)

def quartic(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum(i*x**4)

def quartic_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return i*4*x**3

def schwefel23(x):
    return np.sum(x**10)

def schwefel23_grad(x):
    return 10*x**9

def qing(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum((x**2 - i)**2)

def qing_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return 4*(x**2-i)*x

def sumsquares(x):
    i = np.array(range(1, len(x)+1, 1))
    return np.sum(i*(x**2))

def sumsquares_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    return 2*i*x

def zakharov(x):
    i = np.array(range(1, len(x)+1, 1))
    a = np.sum(x**2)
    b = (.5*np.sum(i*x))**2
    c = (.5*np.sum(i*x))**4
    return a+b+c

def zakharov_grad(x):
    i = np.array(range(1, len(x)+1, 1))
    a = 2*x
    b = np.sum(i*x)*0.5*i
    c = 4*((0.5*np.sum(i*x))**3)*0.5*i
    return a+b+c

def trid(x):
    d = len(x)
    fstar = -d*(d+4)*(d-1)/6.
    return np.sum((x-1)**2) - np.sum(x[1:]*x[:-1]) - fstar

def trid_grad(x):
    xx = np.zeros(len(x)+2)
    xx[1:-1] = x
    return 2*(x-1) - xx[2:] - xx[:-2]

def rosenbrock(x):
    xp = x[1:]
    xm = x[:-1]
    return (100*(xp-xm**2)**2 + (1-xm)**2).sum()

def rosenbrock_grad(x):
    G = np.zeros(len(x))
    G[1:-1] = -400*(x[2:]-x[1:-1]**2)*x[1:-1] - \
                    2*(1-x[1:-1]) + 200*(x[1:-1]-x[:-2]**2)
    G[0] = -400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0])
    G[-1] = 200*(x[-1]-x[-2]**2)
    return G


##############################################################################
if __name__ == '__main__':

    # calling this will generate all the convergence rate plots
    # it will take a while
    
    max_evals = 500 # number of runs of CM and NAG for tuning
    max_evals_rgd = 1000 # number of runs for RGD (this is unfair for RGD)
    
#    x0 = np.array([10,10])
#    v0 = np.zeros(2)
#    pars = {'stepsize': [1e-4, 1e-2],
#             'mu': [.7, .99],
#             'delta': [0, 20],
#             'alpha': [0, 1]}
#    test(x0, v0, booth, booth_grad, 
#         pars, maxiter_tune=150, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=300, fname='booth',
#         markevery=20)
#    
#    x0 = np.array([10, -7])
#    v0 = np.zeros(2)
#    pars = {'stepsize': [1e-4, 1e-1],
#             'mu': [.7, .99],
#             'delta': [0, 15],
#             'alpha': [0, 1]}
#    test(x0, v0, matyas, matyas_grad, 
#         pars, maxiter_tune=200, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=400, fname='matyas',
#         markevery=30)
#    
#    x0 = np.array([10, -10])
#    v0 = np.zeros(2)
#    pars = {'stepsize': [1e-5, 1e-2],
#             'mu': [.8, .99],
#             'delta': [0, 10],
#             'alpha': [0, 1]}
#    test(x0, v0, levi, levi_grad, 
#         pars, maxiter_tune=200, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=400, fname='levi13',
#         markevery=30)
#
#    x0 = 10*np.ones(100)
#    v0 = np.zeros(100)
#    pars = {'stepsize': [1e-6, 1e-2],
#             'mu': [.5, .99],
#             'delta': [0, 10],
#             'alpha': [0, 1]}
#    test(x0, v0, sumsquares, sumsquares_grad, 
#         pars, maxiter_tune=100, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=300, fname='sumsquares',
#         markevery=25)
#    
#    x0 = np.array([-3,-3])
#    v0 = np.zeros(2)
#    pars = {'stepsize': [1e-7, 1e-2],
#             'mu': [.7, .99],
#             'delta': [0, 50],
#             'alpha': [0, 1]}
#    test(x0, v0, beale, beale_grad, pars, maxiter_tune=1000, 
#         max_evals=max_evals, max_evals_rgd=max_evals_rgd, maxiter=2000, 
#         fname='beale', markevery=130)
#    
    x0 = np.array([5, 5])
    v0 = np.zeros(2)
    pars = {'stepsize': [1e-5, 1e-1],
             'mu': [.7, .99],
             'delta': [0, 20],
             'alpha': [0, 1]}
    test(x0, v0, camel, camel_grad, 
         pars, maxiter_tune=100, max_evals=max_evals, 
         max_evals_rgd=max_evals_rgd, maxiter=200, fname='camel',
         markevery=20)
#    
#    x0 = 50*np.ones(50)
#    v0 = np.zeros(50)
#    pars = {'stepsize': [1e-8, 1e-3],
#             'mu': [.7, .99],
#             'delta': [0, 20],
#             'alpha': [0, 1]}
#    test(x0, v0, chung_reynolds, chung_reynolds_grad, 
#         pars, maxiter_tune=1000, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=2000, fname='chung',
#         markevery=150)
#
#    x0 = 2*np.ones(50)
#    v0 = np.zeros(50)
#    pars = {'stepsize': [1e-4, 1e-1],
#             'mu': [.8, .99],
#             'delta': [0, 15],
#             'alpha': [0, 1]}
#    test(x0, v0, quartic, quartic_grad, 
#         pars, maxiter_tune=500, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=1500, fname='quartic',
#         markevery=100)
#
#    x0 = 2.0*np.ones(20)
#    v0 = np.zeros(20)
#    pars = {'stepsize': [1e-6, 1e-2],
#             'mu': [.6, .99],
#             'delta': [0, 40],
#             'alpha': [0, 1]}
#    test(x0, v0, schwefel23, schwefel23_grad, 
#         pars, maxiter_tune=200, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=500, fname='schwefel',
#         markevery=40)
#
#    
#    x0 = 50*np.ones(100)
#    v0 = np.zeros(100)
#    pars = {'stepsize': [1e-6, 1e-2],
#             'mu': [.7, .99],
#             'delta': [0, 20],
#             'alpha': [0, 1]}
#    test(x0, v0, qing, qing_grad, 
#         pars, maxiter_tune=500, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=1000, fname='qing',
#         markevery=50)
#    
#    x0 = 1*np.ones(5)
#    v0 = np.zeros(5)
#    pars = {'stepsize': [1e-7, 1e-2],
#             'mu': [.5, .95],
#             'delta': [0, 15],
#             'alpha': [0, 1]}
#    test(x0, v0, zakharov, zakharov_grad, 
#         pars, maxiter_tune=100, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=200, fname='zakharov',
#         markevery=15)
#
#    d = 1000
#    x0 = 2.048*np.ones(d)
#    x0[int(d/2):] = -2.048
#    v0 = np.zeros(d)
#    pars = {'stepsize': [1e-9, 1e-2],
#             'mu': [.7, .99],
#             'delta': [0, 50],
#             'alpha': [0, 1]}
#    test(x0, v0, rosenbrock, rosenbrock_grad, 
#         pars, maxiter_tune=1000, max_evals=max_evals, 
#         max_evals_rgd=max_evals_rgd, maxiter=2000, fname='rosenbrock',
#         markevery=120)
    
