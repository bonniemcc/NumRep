'''Project Part 1: 
Determine the normalisation functions N1 and N2.
Generate a set of 10000 random events with a
distribution of time and theta given by the PDF.
Plot the time and theta distributions.
'''
import math as m
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour
from pylab import clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize

#Define method to find normalisation function
def N(tau):
    return 3*np.pi*tau*(1-np.exp(-10/(tau)))

#Define method to find PDF
#The first PDF is when F=1 and the second is when F=0
def PDF(time,theta,x0):
    F,tau1,tau2 = x0
    PDF1 = F*(1/N(tau1))*(1+np.cos(theta)**2)*np.exp(-time/tau1)
    PDF2 = (1-F)*(1/N(tau2))*3*(np.sin(theta)**2)*np.exp(-time/tau2)
    return (PDF1 + PDF2)

#Method which uses the box method
#Generates random time and theta values
def random_gen(PDF,x0):
    #Maximum value for the normalised PDF is y2
    #Loop through random values and append if function > y2
    #Set initial values for y and y2
    y = 0.
    y2 = 1.
    while (y2 > y) :
        rand_t = np.random.uniform(0,10)
        rand_theta = np.random.uniform(0,2*m.pi)
        y = PDF(rand_t,rand_theta,x0)
        y2 = np.random.uniform()
    return rand_t, rand_theta


#Define function to plot each PDF with two variables
def plot2D(time,theta,x0,function,equation):
    x,y = meshgrid(time,theta)
    Z = function(x,y,x0)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title(equation)
    plt.show()

#Function to plot histograms of variable distributions
def plothist(variable,xlabel):
    plt.hist(variable,bins=100,density=True)
    plt.title("Normalised histogram")
    plt.xlabel(xlabel)
    plt.ylabel("Normalised Frequency")
    plt.show()


    
def main():

    #Create arrays of two observed quantities
    #Use 100 equally spaced values 
    time = np.arange(0,10,0.1)
    theta = np.arange(0,2*m.pi,2*m.pi/100)

    #Plot both normalised 2D PDFs (F=1,F=0) and total PDF (F=0.5)
    plot2D(time,theta,[1,1,2],PDF,
            '$P_1=(1/N_1)(1+cos^2 theta) exp[-t/ tau_1]$')
    plot2D(time,theta,[0,1,2],PDF,
            '$P_2=(1/N_2)(3sin^2 theta) exp[-t/ tau_2]$')
    plot2D(time,theta,[0.5,1,2],PDF,
            '$P_T=F.P_1 + (F-1).P_2$')

    #Define n (number of time/theta values)
    n = 10000
    #Define parameters (F,tau1,tau2)
    x0 = [0.5,1.0,2.0]

    #Draw time and theta values from PDFs
    time = np.zeros(n)
    theta = np.zeros(n)
    for i in range(n):
        time[i],theta[i] = random_gen(PDF,x0)

    #Plot distribution of time and then theta
    plothist(time,"time")
    plothist(theta,"theta")

main()
