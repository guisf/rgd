"""Optimization solvers: scaled version.

In this version all methods have the mass term removed by an appropriate
rescaling of the momentum variable.

"""

from __future__ import division

import numpy as np


def classical_momentum(x0, v0, function, gradient, maxiter=1000,
                       stepsize=1e-3, mu=0.9):
    """Polyak's heavy ball method."""
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
                    stepsize=1e-3, mu=0.9, delta=0.1):
    """Relativistic method (RGD)."""
    xs = [x0]
    x = x0
    v = v0
    for k in range(maxiter):
        
        g = gradient(x)
        v = mu*v - stepsize*g
        x = x + v/np.sqrt(1 + delta*v.dot(v))

        xs.append(x)

    fs = [function(x) for x in xs]

    return fs, xs

#def relativistic2(x0, v0, function, gradient, maxiter=1000,
#                    stepsize=1e-3, mu=0.9, delta=0.1, alpha=0.5):
#    """Relativistic method, second version.
#    This is based on the leapfrog, but also interpolates between
#    a symplectic and Nesterov's approach as controlled by the
#    parameter alpha:
#        alpha = 0 (Nesterov)
#        alpha = 1 (Leapfrog)
#    
#    """
#    xs = [x0]
#    x = x0
#    v = v0
#    for k in range(maxiter):
#    
#        y = x + mu*v/np.sqrt(1 + (delta*(mu**2))*v.dot(v))
#        g = gradient(y)
#        v = mu*v - stepsize*g
#        x = alpha*y + (1-alpha)*x + v/np.sqrt(1 + delta*v.dot(v))
#
#        xs.append(x)
#
#    fs = [function(x) for x in xs]
#
#    return fs, xs

def relativistic2(x0, v0, function, gradient, maxiter=1000,
                    stepsize=1e-3, mu=0.9, delta=0.1, alpha=1):
    """Relativistic method, second version.
    This is based on the leapfrog, but also interpolates between
    a symplectic and Nesterov's approach as controlled by the
    parameter alpha:
        alpha = 0 (Nesterov)
        alpha = 1 (Leapfrog)
    
    """
    xs = [x0]
    x = x0
    v = v0
    for k in range(maxiter):
    
        x_half = x + np.sqrt(mu)*v/np.sqrt(1 + (delta*mu)*v.dot(v))
        g = gradient(x_half)
        v_half = np.sqrt(mu)*v - stepsize*g
        x = alpha*x_half + (1-alpha)*x + \
                v_half/np.sqrt(1 + delta*v_half.dot(v_half))
        v = np.sqrt(mu)*v_half

        xs.append(x)

    fs = [function(x) for x in xs]

    return fs, xs

def relativistic3(x0, v0, function, gradient, maxiter=1000,
                    stepsize=1e-3, mu=0.9, delta=0.1):
    """Relativistic method, second version.
    This is based on the leapfrog, but also interpolates between
    a symplectic and Nesterov's approach as controlled by the
    parameter alpha:
        alpha = 0 (Nesterov)
        alpha = 1 (Leapfrog)
    
    """
    xs = [x0]
    x = x0
    v = v0
    g = gradient(x)
    for k in range(maxiter):
   
        v = v - stepsize*g
        x = x + v/np.sqrt(1+delta*v.dot(v))
        v = mu*v
        #v = v + (1.-mu)*(1.-1./np.sqrt(1+delta*v.dot(v)))*v
        x = x + v/np.sqrt(1+delta*v.dot(v))
        g = gradient(x)
        v = v - stepsize*g

        xs.append(x)

    fs = [function(x) for x in xs]

    return fs, xs

