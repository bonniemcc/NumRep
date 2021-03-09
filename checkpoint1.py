'''Checkpoint 1: Monte Carlo Simulations - Simulating 500 experiments to measure'''

import numpy as np
import matplotlib.pyplot as plt
#Import class which runs experiment for 1000 muons
from experiment import Experiment
#Import class which repeats experiment 500 times
from repeat import Repeat

def main():
    #The true muon lifetime is 2.2 microseconds
    tau_true = 2.2
    #The number of muons in each experiment is 1000
    num_mu = 1000
    #Use Experiment class to create muons
    muon = Experiment(tau_true,num_mu)
    #Plot data from one experiment
    muon.exp_plot()
    #Use Repeat class to repeat experiment
    muons = Repeat(tau_true)
    #Repeat experiment and plot data for all experiments
    muons.repeat_exp(tau_true,num_mu)
    muons.all_plot()

main()
