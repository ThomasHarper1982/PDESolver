"""
Demonstrate the mixing of 2d and 3d subplots
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

##def f(t):
##    s1 = np.cos(2*np.pi*t)
##    e1 = np.exp(-t)
##    return np.multiply(s1,e1)
##
##
##################
### First subplot
##################
##t1 = np.arange(0.0, 5.0, 0.1)
##t2 = np.arange(0.0, 5.0, 0.02)
##t3 = np.arange(0.0, 2.0, 0.01)
##
### Twice as tall as it is wide.
##fig = plt.figure(figsize=plt.figaspect(2.))
##fig.suptitle('A tale of 2 subplots')
##ax = fig.add_subplot(2, 1, 1)
##l = ax.plot(t1, f(t1), 'bo', 
##            t2, f(t2), 'k--', markerfacecolor='green')
##ax.grid(True)
##ax.set_ylabel('Damped oscillation')
##
##
###################
### Second subplot
###################
##ax = fig.add_subplot(2, 1, 2, projection='3d')
##X = np.arange(-5, 5, 0.25)
##xlen = len(X)
##Y = np.arange(-5, 5, 0.25)
##ylen = len(Y)
##X, Y = np.meshgrid(X, Y)
##R = np.sqrt(X**2 + Y**2)
##Z = np.sin(R)
##
##surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
##        linewidth=0, antialiased=False)
##
##ax.set_zlim3d(-1, 1)
##
##plt.show()

##fig = plt.figure()
##ax = fig.add_subplot(111,projection='3d')
##u=np.linspace(0,2*np.pi,100)
##v = np.linspace(0,np.pi,100)
##
##x=10*np.outer(np.cos(u), np.sin(v))
##y=10*np.outer(np.sin(u), np.sin(v))
##z=10*np.outer(np.ones(np.size(u)), np.cos(v))
##
##ax.plot_surface(x,y,z,rstride=4, cstride=4, color='b')
##
##plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5,5,0.25)
Y = np.arange(-5,5,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
print type(X)
print type(Y)
print type(Z)
surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01,1.01)
ax.w_zaxis.set_major_locator(LinearLocator(10))
ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

fig.colorbar(surf, shrink=0.5,aspect=5)
plt.show()
