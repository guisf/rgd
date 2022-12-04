"""Optimization solvers: non-scaled version.

In this version all methods have a mass term.

"""

from __future__ import division

import numpy as np


def classical_momentum(x0, p0, function, gradient, maxiter=1000,
                       stepsize=1e-3, mu=0.9, mass=1e-3):
    """Polyak's heavy ball method."""
    xs = [x0]
    x = x0
    p = p0
    for k in range(maxiter):

        g = gradient(x)
        p = mu*p - stepsize*g
        x = x + (stepsize/mass)*p
        
        xs.append(x)
    
    fs = [function(x) for x in xs]
    
    return fs, xs

def nesterov(x0, p0, function, gradient, maxiter=1000, 
                stepsize=1e-3, mu=0.9, mass=1e-3/2):
    """Nesterov's accelerated gradient."""
    xs = [x0]
    x = x0
    p = p0
    for k in range(maxiter):

        y = x + mu*(stepsize/2./mass)*p
        g = gradient(y)
        p = mu*p - stepsize*g
        x = x + (stepsize/2./mass)*p
        
        xs.append(x)
    
    fs = [function(x) for x in xs]
    
    return fs, xs

def relativistic(x0, p0, function, gradient, maxiter=1000, 
                    stepsize=1e-3, mu=0.9, c=1e4, mass=1e-3):
    """Relativistic method (RGD)."""
    xs = [x0]
    x = x0
    p = p0
    for k in range(maxiter):
        
        g = gradient(x)
        p = mu*p - stepsize*g
        x = x + stepsize*c*p/np.sqrt((mass*c)**2 + p.dot(p))

        xs.append(x)

    fs = [function(x) for x in xs]

    return fs, xs

def relativistic2(x0, p0, function, gradient, maxiter=1000,
                    stepsize=1e-3, mu=0.9, c=1e4, mass=1e-3, alpha=0.5):
    """Relativistic method, second version.
    This is based on the leapfrog, but also interpolates between
    a symplectic and Nesterov's approach as controlled by the
    parameter alpha:
        alpha = 0 (Nesterov)
        alpha = 1 (Leapfrog)
    
    """
    xs = [x0]
    x = x0
    p = p0
    for k in range(maxiter):
    
        y = x + (stepsize/2.)*mu*c*p/np.sqrt((mass*c)**2 + p.dot(p))
        g = gradient(y)
        p = mu*p - stepsize*g
        x = alpha*y + (1-alpha)*x + \
                (stepsize/2.)*c*p/np.sqrt((mass*c)**2 + p.dot(p))

        xs.append(x)

    fs = [function(x) for x in xs]

    return fs, xs


