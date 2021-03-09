'''Checkpoint 3: Solving First Order ODEs for the PN Junction
    using the Euler and 4th Order Runge-Kutta (RK4) methods''' 

import math as m
import numpy as np
import matplotlib.pyplot as plt
#Use Class: ChargeDistribution.py
from ChargeDistribution import ChargeDistribution
    #Has two useful methods: q=evaluate(x) and plot show()

#Define useful methods

#Method for finding charge values for set x values
#Using class method evaluate(x)
def Charge(x_vals,data):
    rho_vals = np.zeros(len(x_vals))
    for i in range(len(x_vals)):
        rho_vals[i] = data.evaluate(x_vals[i])
    return rho_vals

#Euler method function
def Euler(x_vals,delta,data,y,variable):
    y_vals = np.zeros(len(x_vals))
    if variable == "a":
        for i in range(len(x_vals)-1):
            y_vals[i+1] = y_vals[i] + delta*data.evaluate(x_vals[i])
    if variable == "b":
        for i in range(len(x_vals)-1):
            y_vals[i+1] = y_vals[i] + delta*y[i]
    return y_vals
    
#RK4 method function
def RK4(x_vals,delta,data,new_y,variable):
    z_vals = np.zeros(len(x_vals))
    if variable == "a":
        for i in range(len(x_vals)-1):
            k1 = data.evaluate(x_vals[i])
            k2 = data.evaluate(x_vals[i]+delta/2)
            k3 = k2
            k4 = data.evaluate(x_vals[i]+delta)
            z_vals[i+1] = z_vals[i] + delta*(k1/6+k2/3+k3/3+k4/6)
    if variable == "b":
        for i in range(len(x_vals)-1):
            k1 = new_y[2*i]
            k2 = new_y[2*i-1]
            k3 = k2
            k4 = new_y[2*(i+1)]
            z_vals[i+1] = z_vals[i] + delta*(k1/6+k2/3+k3/3+k4/6)
    return z_vals

#Plot function for Euler and RK4 Method
def plot(x_vals,y_vals,z_vals,title):
    plt.plot(x_vals,y_vals, x_vals, z_vals, 'r')
    plt.legend(["Euler method","RK4 method"])
    plt.xlabel("x")
    plt.ylabel(str(title))
    plt.title(str(title)+ " Distribution")
    plt.show()


def main():
    #Initialise class data
    data = ChargeDistribution()
    
    #Choose delta and set x values between -2 and 2
    delta = float( input( "Enter step size (delta) "))
    x_vals = np.arange(-2,2+delta,delta)

    #Part 0: Calculate charge values and plot charge distribution
    Q = Charge(x_vals,data)
    data.show('Charge Distribution')
    
    #Part 1: Determine E(x) in the range [-2,2]
    
    #Implement Euler method
    y1 = Euler(x_vals,delta,data,1,"a")
    #Implement RK4 method
    z1 = RK4(x_vals,delta,data,1,"a")
    #Plot E(x) for both methods
    plot(x_vals,y1,z1,"Electric Field")

    #Part 2: Determine Voltage in the range [-2,2]
    
    #Need to calculate new E values for twice as many x values
    new_x = np.arange(-2,2+delta,delta/2)
    #new_q = Charge(new_x,data)
    #Find new E values using Euler method
    #Step size is half the size of delta
    new_y = Euler(new_x,delta/2,data,1,"a")
    
    #Implement Euler method
    y2 = Euler(x_vals,delta,data,y1,"b")
    #Implement RK4 method
    z2 = RK4(x_vals,delta,data,new_y,"b")
    #Plot V(x) for both methods
    plot(x_vals,-y2,-z2,"Voltage")
    
main()
