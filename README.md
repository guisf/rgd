# Relativistic Gradient Descent (RGD)

RGD is an optimization method based on the simulation of a relativistic particle under the influence of a potential (objective function) and friction. We use a symplectic integrator to simulate the system. Gradient descent (GD) is a very well known method. The classical momentum method (CM), also known as olyak's heavy ball, and Nesterov's accelerated gradient method (NAG) are accelerated variants of GD, often used in machine learning.
RGD generalizes both CM and NAG, and have usually a superior performance. For instance, its convergence rate in a matrix completion problem (which is nonconvex) is illustrated below:

![This is an image](https://myoctocat.com/assets/images/base-octocat.svg)
 
This method was proposed in the G. França et. al., "Conformal symplectic and relativistic optimization,"  J. Stat. Mech. (2020) 124008 (https://iopscience.iop.org/article/10.1088/1742-5468/abcaee)

A shorter version of this paper was also published at NeurIPS 2020 (spotlight):
https://proceedings.neurips.cc/paper/2020/hash/c4b108f53550f1d5967305a9a8140ddd-Abstract.html
