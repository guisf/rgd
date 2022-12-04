import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


def func(x):
    a, b = x
    return (1.5-a+a*b)**2 + (2.25-a+a*(b**2))**2+(2.625-a+a*(b**3))**2

xran = [-4, 4, 0.05]
yran = [-4, 4, 0.05]
title = 'Function'

fig = plt.figure(figsize=(2*5, 5))
ax0 = fig.add_subplot(121, projection='3d')

ax0.xaxis.pane.fill = False
ax0.yaxis.pane.fill = False
ax0.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax0.xaxis.pane.set_edgecolor('w')
ax0.yaxis.pane.set_edgecolor('w')
ax0.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
ax0.grid(False)

x = np.arange(*xran)
y = np.arange(*yran)
X, Y = np.meshgrid(x, y)
zs = np.array([func(np.array([x,y]))
               for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
#my_col = plt.cm.jet_r(Z/np.amax(Z))
#ax0.plot_surface(X, Y, Z, rstride=10, cstride=10, facecolors=my_col,
#        linewidth=0, antialiased=False)
ax0.plot_surface(X, Y, Z, 
        #rstride=1, cstride=1, 
        #cmap='jet_r',
        #linewidth=0, 
        alpha=.2,
        shade=True,
        antialiased=False)
xran = [1, 4, 0.01]
yran = [-0.5, 1.5, 0.01]
x = np.arange(*xran)
y = np.arange(*yran)
X, Y = np.meshgrid(x, y)
zs = np.array([func(np.array([x,y]))
               for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax0.plot_surface(X, Y, Z, color='red')
#ax0.plot_surface(X,Y,Z,cmap='jet_r',
                 #linewidth=0.3,
                 #rstride=10, cstride=10,
#                 alpha=.7,
                 #edgecolor='k',
                 #norm=norm,
#                 shade=True
#                 )
            #linewidths=1,linestyles='solid',
#            cmap='binary')
#ax0.contour(X,Y,Z,zdir ='x',offset=0,
            #linewidths=1,linestyles='solid',
#            cmap='binary')
#ax0.contour(X,Y,Z,zdir='y',offset=0,
            #linewidths=1,linestyles='solid',
#            cmap='binary')
ax0.set_xlabel(r'$x$')
ax0.set_ylabel(r'$y$')
#ax0.set_zlabel(r'$f(x)$')
ax0.set_title(title)
ax0.view_init(40, 30)

ax = fig.add_subplot(122)
ax.contourf(X, Y, Z, 1000)

fig.savefig('test_surface.pdf')
