# Relativistic Gradient Descent (RGD)

RGD is a simple optimization method based on the simulation of a relativistic particle under the influence of a potential (objective function) and friction. We use a symplectic integrator to simulate such a physical system. 

Gradient descent (GD) is probably the most well-known optimization method. The classical momentum method (CM), also known as Polyak's heavy ball, and Nesterov's accelerated gradient method (NAG) are accelerated variants of GD which are extensively used in machine learning.
RGD generalizes both CM and NAG and usually have a superior performance. For instance, its convergence rate in a matrix completion problem (which is nonconvex) is illustrated in the figure below.

![](https://github.com/guisf/rgd/blob/main/figs/mat_comp_rate.png)
 
* This method was proposed in the [G. França et. al., "Conformal symplectic and relativistic optimization,"  J. Stat. Mech. (2020) 124008](https://iopscience.iop.org/article/10.1088/1742-5468/abcaee).
* A shorter version of this paper was also published at [NeurIPS 2020 (spotlight)](https://proceedings.neurips.cc/paper/2020/hash/c4b108f53550f1d5967305a9a8140ddd-Abstract.html).
* Besides the above papers, see the [Presentation](https://github.com/guisf/rgd/blob/main/Franca_talk_NeurIPS2020.pdf) for a quick introduction or the [Poster](https://github.com/guisf/rgd/blob/main/poster_franca.pdf) for an even quicker one.
