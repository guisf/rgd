"""
Test suite for different objective functions.

See https://arxiv.org/pdf/1308.4008.pdf

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from opt import OPT
import solvers_scaled as solvers1

from test_suite import *

def plot_surface(func, xran=[1,5,0.01], yran=[0,1.5,0.01], 
                 fname='name', a=None, b=None, vmin=None, vmax=None):
    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')
    x = np.arange(*xran)
    y = np.arange(*yran)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func(np.array([x,y])) 
                   for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    if vmin is not None and vmax is not None:
        ax.plot_surface(X,Y,Z,cmap='jet_r',vmin=vmin,vmax=vmax)
    else:
        ax.plot_surface(X,Y,Z,cmap='jet_r')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$f$')
    if a is not None and b is not None:
        ax.view_init(a, b)
    plt.tight_layout()
    #fig.savefig('figs/%s_surface.pdf'%fname)
    fig.savefig('figs/%s_surface.png'%fname)

def plot_contour(func, xran=[1,5,0.01], yran=[0,1.5,0.01],
                 vmin=0, vmax=10, fname='name'):
    fig = plt.figure()
    ax = fig.add_subplot('111')
    x = np.arange(*xran)
    y = np.arange(*yran)
    X, Y = np.meshgrid(x, y)
    zs = np.array([func(np.array([x,y])) 
                   for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    cs = ax.imshow(Z,origin='lower',cmap='inferno_r',
            extent=[X.min(),X.max(),Y.min(),Y.max()],
            vmin=vmin,vmax=vmax,
            aspect='auto'
    )
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.tight_layout()
    plt.savefig('figs/%s_contour.pdf'%fname, bbox_inches='tight')


##############################################################################
if __name__ == '__main__':
    
    plot_surface(booth,xran=[-5,5,0.01],yran=[-2,8,0.01],fname='booth',vmin=0,vmax=100)
    plot_surface(matyas,xran=[-9,9,0.01],yran=[-9,9,0.01],fname='matyas',a=45,b=20,vmin=0,vmax=10)
    plot_surface(levi,xran=[-9,10,0.05],yran=[-9,10,0.05],fname='levi',a=50,b=140,vmin=0,vmax=200)
    plot_surface(sumsquares,xran=[-9,9,0.05],yran=[-9,9,0.05],a=45,b=30,vmin=0,vmax=100,fname='sumsquares')
    plot_surface(beale,xran=[-3.5,3.5,0.01],yran=[-3.5,3.5,0.01],fname='beale',a=45,b=45,vmin=0,vmax=200)
    plot_surface(chung_reynolds,xran=[-50,50,0.1],yran=[-50,50,0.1],vmin=0,vmax=5e6,a=45,b=30,fname='chung')
    plot_surface(camel,xran=[-2.5,2.5,0.01],yran=[-2.5,2.5,0.01],a=40,b=115,vmin=0,vmax=8,fname='camel')
    plot_surface(quartic,xran=[-1.8,1.8,0.01],yran=[-1.8,1.8,0.01],a=50,b=30,vmin=0,vmax=10,fname='quartic')
    plot_surface(schwefel23,xran=[-2,2,0.01],yran=[-2,2,0.01],a=50,b=30,vmin=0,vmax=1000,fname='schwefel')
    plot_surface(qing,xran=[-4,4,0.05],yran=[-4,4,0.05],a=60,b=30,vmin=0,vmax=30,fname='qing')
    plot_surface(zakharov,xran=[-2,2,0.05],yran=[-2,2,0.05],vmin=0,vmax=10,a=45,b=30,fname='zakharov')
    plot_surface(rosenbrock,xran=[-2.1,2.1,0.05],yran=[-2,3,0.05],vmin=0,vmax=700,a=45,b=230,fname='rosenbrock')


