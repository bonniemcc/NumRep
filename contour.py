import math as m
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# the function that I'm going to plot
def func(t,z):
 return exp(z)+(t)

#(1+m.cos(z)*m.cos(z))*m.exp(-t)

n = 10

t = arange(0,10,10/n)
z = arange(0,2*m.pi,2*m.pi/n)
T,Z = meshgrid(t, z) # grid of point
P = func(T, Z) # evaluation of the function on the grid

im = imshow(P,cmap=cm.RdBu) # drawing the function
# adding the Contour lines with labels
cset = contour(P,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
#show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import sin,sqrt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T, Z, P, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
