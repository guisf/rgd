"""Interface to solve optimization problems."""

import numpy as np
from hyperopt import hp, tpe, Trials, fmin

import solvers_scaled as solvers1
import solvers_mass as solvers2


class OPT:
    """Optimization Problem class."""

    def __init__(self, objective_function, gradient, x0, v0):
        """
        objective_function: accepts vector and returns a real.
        gradient: gradient of the objective function.
        x0, v0: initial state
        
        """
        self.f = objective_function
        self.g = gradient
        self.x0 = x0
        self.v0 = v0
        self.params = {}
        self.best = {}

    def set_solver(self, solver):
        """solver: function that implements an optimization algorithm.
        This function will be called in other methods.
        
        """
        self.solver = solver
    
    def solve(self, maxiter=1000, **params):
        """Call the solver.
        
        x0, v0: initial state
        maxiter: maximum number of iterations
        params: keyword arguments for the solver function.
        
        """
        if params:
            #print("Using input parameters.")
            p = params
        elif self.params:
            #print("Using internal parameters.")
            p = self.params
        elif self.best:
            #print("Using previously tuned parameters.")
            p = self.best 
        else:
            raise ValueError('No parameters for the optimization solver.')
        self.scores, self.states = self.solver(self.x0, self.v0, 
                                        self.f, self.g, maxiter=maxiter, **p)

    def tune(self, param_range, maxiter=1000, max_evals=100):
        """Tune the algorithm with Bayesian optimization."""
        
        def tune_objective(theta):
            # this function must accept a single argument (parameters)
            # and returns a score
            self.solve(maxiter=1000, **theta)
            return self.scores[-1]

        space = {}  # range for parameter search
        for par_name, par_range in param_range.items():
            space[par_name] = hp.uniform(par_name, *par_range)

        tpe_algo = tpe.suggest
        tpe_trials = Trials()
        tpe_best = fmin(fn=tune_objective, space=space, algo=tpe_algo, 
                        trials=tpe_trials, max_evals=max_evals)
        self.tpe_trials = tpe_trials
        self.best = tpe_best
            

