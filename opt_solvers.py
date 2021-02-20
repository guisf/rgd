# Optimization solvers
# Classical momentum, Nesterov, and Relativistic Gradient Descent
#
# Guilherme  S. Franca <guifranca@gmail.com>

import numpy as np


def classical_momentum(x0, gradient, maxiter=1000, stepsize=1e-3, mu=0.9):
    """Polyak's heavy ball method, also  known as classical momentum."""
    
    x = x0
    v = np.zeros_like(x0)
    xs = [x]
    for k in range(maxiter):
        g = gradient(x)
        v = mu*v - stepsize*g
        x = x + v 
        xs.append(x)
    return xs

def nesterov(x0, gradient, maxiter=1000, stepsize=1e-3, mu=0.9):
    """Nesterov's accelerated gradient."""

    x = x0
    v = np.zeros_like(x0)
    xs = [x]
    for k in range(maxiter):
        y = x + mu*v
        g = gradient(y)
        v = mu*v - stepsize*g
        x = x + v
        xs.append(x)
    return xs

def relativistic(x0, gradient, maxiter=1000,
                 stepsize=1e-3, mu=0.9, delta=0.1, alpha=1):
    """Relativistic method, based on dissipative version of the leapfrog."""

    x = x0
    v = np.zeros_like(x0)
    xs = [x]
    for k in range(maxiter):
        x_half = x + np.sqrt(mu)*v/np.sqrt(1+(delta*mu)*v.dot(v))
        g = gradient(x_half)
        v_half = np.sqrt(mu)*v - stepsize*g
        x = alpha*x_half + (1-alpha)*x + \
            v_half/np.sqrt(1 + delta*v_half.dot(v_half))
        v = np.sqrt(mu)*v_half
        xs.append(x)
    return xs

