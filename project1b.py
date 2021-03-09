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

#Function to integrate PDF
def integrate(function,a,b,c,d,args):
    return dblquad(function,a,b,c,d,args)

#Function to plot PDFs
def plotPDF(time,theta,F,N,tau,func,equation):
    plt.hist(func(time,theta,F,N,tau),bins=50,density=True)
    plt.title(equation)
    plt.xlabel(" ")
    plt.ylabel("Normalised Frequency")
    plt.show()

    
def main():
     
    #Set parameters
    n = 10000
    F = 0.5
    tau = [1.0,2.0]
    #Set N1 and N2 to 1 initially (normalise later)
    N = [1,1]
    #Create lists of two observed quantities using random numbers
    time = np.random.uniform(0,10,n)
    theta = np.random.uniform(0,2*m.pi,n)
    
    #Integrate each PDF with N=1 to find normalisation values N1 and N2
    N1 = integrate(PDF,0,2*m.pi,0,10,[1,N,tau])
    N2 = integrate(PDF,0,2*m.pi,0,10,[0,N,tau])

    #Set new normalisation values
    N[0] = N1[0]
    N[1] = N2[0]

    #Plot both normalised PDFs (F=1,F=0) and total PDF (F=0.5)
    #First plot while holding theta constant and varying time
    theta_const = np.full(n,m.pi)
    plotPDF(time,theta_const,1,N,tau,PDF,
            '$P_1=(1/N_1)(1+cos^2 theta) exp[-t/ tau_1]$')
    plotPDF(time,theta_const,0,N,tau,PDF,
            '$P_2=(1/N_2)(3sin^2 theta) exp[-t/ tau_2]$')
    plotPDF(time,theta_const,0.5,N,tau,PDF,
            '$P_T=F.P_1 + (F-1).P_2$')
    #Then plot while holding time constant and varying theta
    time_const = np.zeros(n)
    plotPDF(time_const,theta,1,N,tau,PDF,
            '$P_1=(1/N_1)(1+cos^2 theta) exp[-t/ tau_1]$')
    plotPDF(time_const,theta,0,N,tau,PDF,
            '$P_2=(1/N_2)(3sin^2 theta) exp[-t/ tau_2]$')
    plotPDF(time_const,theta,0.5,N,tau,PDF,
            '$P_T=F.P_1 + (F-1).P_2$')

main()
