'''Project Part 1: 
Determine normalisation functions N1(tau1) and N2(tau2).
Generate 10,000 random events with time and theta drawn from the PDF.
Plot the t and θ distributions of the data you have generated.
Comment on how easily (or not) a fit would be able to distinguish the
components from each distribution alone.
'''

import math as m
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.integrate import quad, dblquad

#Define useful functions

#Define function to find PDF
#The PDF of the decay is given by a fraction F of P1 and the remainder of P2
#The first PDF is when F=1 and the second is when F=0
def PDF(time,theta,F,N,tau):
    return F*(1/N[0])*(1+np.cos(theta)*np.cos(theta))*np.exp(-time/tau[0]) + (1-F)*(1/N[1])*3*np.sin(theta)*np.sin(theta)*np.exp(-time/tau[1])

#Define function to plot each PDF with two variables
def plotPDF(time,theta,F,N,tau,function,equation):
    x,y = meshgrid(time,theta)
    Z = function(x,y,F,N,tau)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title(equation)
    plt.show()

#Function to integrate PDF
def integrate(function,a,b,c,d,args):
    return dblquad(function,a,b,c,d,args)
    
def main():
     
    #Set parameters
    n = int(input("Enter number of values of time and theta: " ))
    F = 0.5
    tau = [1.0,2.0]
    #Set N1 and N2 to 1 initially (normalise later)
    N = [1,1]
    #params=[N,tau]
    #Create lists of two observed quantities
    time = np.arange(0,10,10/n)
    theta = np.arange(0,2*m.pi,2*m.pi/n)
    
    #Integrate each PDF with N=1 to find normalisation values N1 and N2
    N1 = integrate(PDF,0,2*m.pi,0,10,[1,N,tau])
    N2 = integrate(PDF,0,2*m.pi,0,10,[0,N,tau])

    #Set new normalisation values
    N[0] = N1[0]
    N[1] = N2[0]

    #Plot both normalised PDFs (F=1,F=0) and total PDF (F=0.5)
    plotPDF(time,theta,1,N,tau,PDF,
            '$P_1=(1/N_1)(1+cos^2 theta) exp[-t/ tau_1]$')
    plotPDF(time,theta,0,N,tau,PDF,
            '$P_2=(1/N_2)(3sin^2 theta) exp[-t/ tau_2]$')
    plotPDF(time,theta,0.5,N,tau,PDF,
            '$P_T=F.P_1 + (F-1).P_2$')

main()


