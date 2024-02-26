#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 11 12:11:19 2023

@author: adrianharkness
"""

import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from scipy.optimize import minimize
import time

# Set the dpi value for high-definition plots
plt.rcParams['figure.dpi']=300

class NoisyEntanglementDistribution:
    '''
    This class returns the probabilities associated with each of the four bell
    states being the final state of the honest network after entanglement 
    distribution (P0, P1, P2, P3).
    '''

    def __init__(self, size, channel_noise):
        #size is the number of channels in the network
        #channel_noise is a list of the probabilities of each bell state in each channel
        self.size = size
        self.channel_noise = channel_noise
        p0, p1, p2, p3 = channel_noise

        # Bell State Measurement lookup table
        BSM = np.array([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]])

        # list of 4^n possible strings describing bell states in each of the n
        # channels before BSM measurements.  Ranges from 000...0 to 333...3
        InitialStateList = list(product((0, 1, 2, 3), repeat=self.size))
        
        # list of probabilities associated with each state in the state list
        ProbabilityStateList = []
        for state in InitialStateList:
            a = state.count(0)
            b = state.count(1)
            c = state.count(2)
            d = state.count(3)
            probability = (p0)**a * (p1)**b * (p2)**c * (p3)**d
            ProbabilityStateList.append(probability)

        self.ProbabilityStateList = np.array(ProbabilityStateList)

        FinalStateList = []
        # for each possible combination of states,
        # find resulting bell state after all BSMs
        StateList = InitialStateList.copy()
        for index in range(len(StateList)):
            state = StateList[index]
            state = list(state)
            while len(state) > 1:
                # consecutive Bell State Measurements from left to right until
                # just one bell state
                leftbellstate = state[0]
                rightbellstate = state[1]
                measurementresult = BSM[leftbellstate, rightbellstate]
                state[0] = measurementresult
                del state[1]
            FinalStateList.append(state[0])

        self.FinalStateList = np.array(FinalStateList)

        # Initialize dictionaries to store probabilities for each state
        state_probabilities = {}

        # Loop through unique states in FinalStateList
        for state in np.unique(self.FinalStateList):
            # Find indices for the current state
            indices = np.where(self.FinalStateList == state)[0]

            # Calculate probabilities for the current state
            probabilities = self.ProbabilityStateList[indices]

            # Calculate the sum of probabilities for the current state
            state_probabilities[f'P{state}'] = sum(probabilities)

        # Assign the calculated probabilities to instance variables
        for state, probability in state_probabilities.items():
            setattr(self, state, probability)

        self.states = pd.DataFrame({'Initial State': InitialStateList,
                                    'Final State': self.FinalStateList,
                                    'Probability': self.ProbabilityStateList})

class network:
    '''
    Returns Qx and P* as functions of the total network size, 
    the honest network size, and the link-level depolarization noise.
    Qx is the error rate of the full network, P* is the error rate of the honest network.
    '''

    def __init__(self, full_size, honest_size, channel_noise):
        if honest_size > full_size:
            raise ValueError("Honest size cannot be greater than full size")
        
        self.full_size = full_size
        self.honest_size = honest_size
        self.channel_noise = channel_noise

        # P1 + P3 (x basis / phase flip noise) of whole network
        full = NoisyEntanglementDistribution(
            self.full_size, self.channel_noise)
        #fix numerical errors if Qx is greater than 0.5
        self.Qx = min(.5, full.P1 + full.P3)

        if honest_size == 0:
            self.pstar = 0
            self.states = pd.DataFrame({'states': [full.P0, full.P1, full.P2, full.P3]})
        else:
            # P1 + P3 (x basis / phase flip noise) of honest network only
            honest = NoisyEntanglementDistribution(
                self.honest_size, self.channel_noise)
            #fix numerical errors if pstar is greater than 0.5
            self.pstar = min(.5, honest.P1 + honest.P3)
            self.states = pd.DataFrame({'full': [full.P0, full.P1, full.P2, full.P3],
                                        'honest': [honest.P0, honest.P1, honest.P2, honest.P3]
                                        })
      
# Depolarization model: P(phi_i) = Pi = (1-3Q/4)phi0 + (Q/4)(phi1+phi2+phi3)
#returns p0, p1, p2, p3
def depolarization(q):
    return [(1-3*q/4), q/4, q/4, q/4]

#modified binary entropy function (h-bar)
def bin_entropy(p):
    #prevents log_2 errors at p=0
    def safe_log(p):
        p = np.where(p < 0.0000001, 1, p)
        return np.log2(p)
    return np.where(p < 0.5, -p*safe_log(p) - (1-p)*safe_log(1-p), 1)

#Noise at which keyrate is 0
def noise_tolerance(keyrates, noises):
    # find the noise tolerance for a given key rate
    index = keyrates.index(0)
    tolerance = noises[index]
    return tolerance

def plot_network_noise_vs_channel_noise(full_size, honest_size):
    # define q values
    q_values = np.linspace(0, 1, 100)

    # initialize lists to store Qx and pstar values
    Qx_values = []
    pstar_values = []

    # loop through q values and calculate Qx and pstar for each
    for q in q_values:
        # create network object
        network_obj = network(full_size, honest_size, depolarization(q))
        # append Qx and pstar values to lists
        Qx_values.append(network_obj.Qx)
        pstar_values.append(network_obj.pstar)

    # plot Qx and pstar as a function of channel noise
    plt.plot(q_values, Qx_values, label='Measured Noise $Qx$')
    plt.plot(q_values, pstar_values, label='Natural Depolarization $P*$')
    plt.grid(True)
    plt.ylim(0, .6)
    plt.xlabel('Channel Depolarization $q$')
    plt.ylabel('Network Noise')
    plt.title(f'{full_size} total links, {honest_size} honest links')
    plt.legend()
    plt.show()

#finite-key rate for a noisy partially corrupted network (eq 67)
def keyrate(N, Qx, pstar, epsilon=1e-36):
    '''
    Theorem 2 / eq. 39
    N total number of signals transmitted (rounds of communication)
    Qx error rate in full network
    pstar error rate in honest network
    '''
    #number of signals in measured test set
    #should I round to nearest integer?
    m = np.round(.07*N)
    #number of signals remaining for secret key
    n = N-m
    #ideal state delta
    delta = np.sqrt((m+n+2)*np.log(2/(epsilon**2)) / (m*(m+n)))
    #ideal-ideal state delta
    delta_prime = np.sqrt(np.log(2/epsilon) / (2*m))
   
    #keyrate evaluation
    #Do I need hamming weight of Qx? Qx isn't a binary string here...
    if pstar == 0.5:
        keyrate = (-1.2*(bin_entropy(Qx))) / N
    else:
        #Q = ((Qx - pstar + delta_prime)/(1-2*pstar)) + delta #39
        #keyrate = (n/N)*(1 - bin_entropy(Q) - (bin_entropy(Qx))) #39
        Q = ((Qx - pstar + delta_prime)/(1-2*pstar)) #67
        keyrate = (n/N)*(1 - bin_entropy(Q)) - 1.2*(bin_entropy(Qx)) - (np.log2(1/epsilon))/N #67
    
    #bound keyrates to be between 0 and 1
    #keyrate = max(0, keyrate)
    if keyrate > 1:
        keyrate = 1
    return keyrate

#asymptotic key rate (eq 68)
def keyrate_inf(Qx, pstar):
   
    #keyrate evaluation
    if pstar == 0.5:
        keyrate = (-1.2*(bin_entropy(Qx)))
    else:
        Q = (Qx - pstar)/(1-2*pstar)
        keyrate = 1 - bin_entropy(Q) - bin_entropy(Qx)
    #bound keyrates to be between 0 and 1
    #keyrate = max(0, keyrate)
    if keyrate > 1:
        keyrate = 1
    return keyrate

def finite_BB84_keyrate_2(N, Q, epsilon=1e-36):
    """
    Quantum Sampling for Finite-Key Rates in High Dimensional Quantum Cryptography
    Krawec Et. Al. 2022
    Eq. 36
    """
    def single_rate(N,Q,epsilon): 
        #number of signals in measured test set
        m = np.round(.07*N)
        #number of signals remaining for secret key
        n = N-m
        
        def gamma(x):
            alpha, beta, nu, zeta = x[0], x[1], x[2], x[3]
            return (1/((m+m)*(Q+zeta)+1)) + (1/(n+m-(n+m)*(Q+zeta)+1))
        
        def f(x):
            alpha, beta, nu, zeta = x[0], x[1], x[2], x[3]
            return np.sqrt(np.exp(-(2*(n+m)*m*(zeta**2))/(n+1)) + np.exp((-2*gamma(x))*(((n*(nu-zeta))**2)-1)))

        def g(x):
            alpha, beta, nu, zeta = x[0], x[1], x[2], x[3]
            return np.exp(-(n*(m**2)*(nu**2))/((n+m)*(m+1)))

        #objective function
        fun = lambda x: -x[0]

        #constraints
        cons = ({'type': 'ineq', 'fun': lambda x: x[2] -  x[3]},
                {'type': 'ineq', 'fun': lambda x: epsilon - (epsilon/100 + 2*f(x) + g(x))})
        
        #bounds
        bnds = ((0, 1), (0, .5), (0, max(.5-Q, 0)), (0, max(.5-Q, 0)))

        #initial guess
        x0 = np.array([0, 1/3, 0, .1])

        keyrate = minimize(fun, x0, bounds=bnds, constraints=cons, options={'disp': True})
        
        return keyrate
    
    Q = np.asarray(Q)
    keyrates = np.vectorize(single_rate)(N, Q, epsilon)
    return keyrates

def finite_BB84_keyrate(N, Q, epsilon=1e-36):
    """
    Quantum Sampling for Finite-Key Rates in High Dimensional Quantum Cryptography
    Krawec Et. Al. 2022
    eq. 34
    """
    m = np.round(.07*N)
    n = N-m
    #dimension
    d = 2
    #confidence interval?
    nu = np.sqrt(((n+m)*(m+1)*np.log(2/epsilon))/((m**2)*n))
    #keyrate
    keyrate = (n/N)*(np.log2(d) - 2*bin_entropy(Q + nu) - (Q + nu)*np.log2(d - 1))
    #keyrate = (n/N)*(1-bin_entropy(Q + nu))
    return keyrate

def BB84_keyrate(noise):
    # Check if noise is a list
    if isinstance(noise, np.ndarray) or isinstance(noise, list):
        # Calculate keyrate for each noise value
        keyrates = [1 - 2*bin_entropy(n) for n in noise]
        # Bound keyrates between 0 and 1
        keyrates = [max(0, min(kr, 1)) for kr in keyrates]
        return keyrates
    else:
        # Calculate keyrate for a single noise value
        keyrate = 1 - 2*bin_entropy(noise)
        # Bound keyrate between 0 and 1
        keyrate = max(0, min(keyrate, 1))
        return keyrate

#finite key
def plot_keyrate_vs_signalrounds(q, full_size, honest_sizes):
    # define range of signal rounds
    signal_rounds = np.logspace(4, 9, num=10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        # network object
        network_obj = network(full_size, honest_size, depolarization(q))
        # initialize list to store key rates
        keyrates = []
        # loop through signal rounds and calculate key rate for each
        for signal in signal_rounds:
            keyrate_val = keyrate(signal, network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_val)
        print(f"Network: {honest_size}/{full_size}")
        # plot key rate as a function of signal rounds
        plt.plot(signal_rounds, keyrates, label=f'Honest links: {honest_size}')
    #BB84
    BB84_keyrates = []
    network_obj = network(full_size, 0, depolarization(q))
    print(f"Network Noise: {network_obj.Qx}")
    for signal in signal_rounds:
        keyrate_val = finite_BB84_keyrate(signal, network_obj.Qx)
        BB84_keyrates.append(keyrate_val)
    plt.plot(signal_rounds, BB84_keyrates, color='black', label='BB84-F', linestyle='dotted', linewidth=2)

    #plt.grid(True)
    plt.xscale('log')
    plt.yscale('linear')
    #plt.ylim(0, .3)
    plt.xlim(1e5, 1e9)
    plt.xlabel('Number of Signals $N$')
    plt.ylabel('Key-Rate')
    #plt.title(f'{full_size} total links, {q*100}% channel depolarization')
    plt.legend()
    plt.show()

#finite key
def plot_keyrate_vs_Qx(full_size, honest_sizes, N):
    # define q values
    q_values = np.linspace(0, 1, 10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        #initialize lists to store Qx values
        Qx_values = []
        # initialize list to store key rates
        keyrates = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # append Qx values to list
            Qx_values.append(network_obj.Qx)
            keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_val)
        print(f"Network: {honest_size}/{full_size}")
        # plot key rate as a function of Qx
        plt.plot(Qx_values, keyrates, label=f'Honest links: {honest_size}')
    plt.plot(np.linspace(0,.5,1000), finite_BB84_keyrate(N, np.linspace(0,.5,1000)), label='BB84-F', color='black', linestyle='dotted', linewidth=2)
    #plt.grid(True)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlim(0,.2)
    plt.xlabel('Noise $Q_x$')
    plt.ylabel('Key-Rate')
    #plt.title(f'Finite-Key Rates for {full_size} Total Links, {N:.1e} Signal Rounds')
    plt.legend()

    # Create inset of width 30% and height 30% of the parent axes' bounding box at the lower left corner (at 0.05, 0.1)
    axins = plt.gca().inset_axes([.05, .1, 0.3, 0.3])
    for honest_size in honest_sizes:
        Qx_values = []
        keyrates = []
        for q in q_values:
            network_obj = network(full_size, honest_size, depolarization(q))
            Qx_values.append(network_obj.Qx)
            keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_val)
        axins.plot(Qx_values, keyrates, label=f'Honest links: {honest_size}')
    axins.plot(np.linspace(0,.5,1000), finite_BB84_keyrate(N, np.linspace(0,.5,1000)), label='BB84-F', color='black', linestyle='dotted', linewidth=2)
    axins.set_xlim(0, .05)  # apply the x-limits
    axins.set_ylim(.2, 1)  # apply the y-limits
    axins.set_xscale('linear')
    axins.set_yscale('log')
    axins.set_xticklabels([])  # remove xtick labels
    axins.set_yticks([])  # remove yticks
    axins.set_yticklabels([], minor=True)  # remove ytick labels

    # Add rectangle and connecting lines from the rectangle to the inset axes
    plt.gca().indicate_inset_zoom(axins, edgecolor="black")

    plt.show()

def plot_asymptotic_keyrate_vs_Qx(full_size, honest_sizes):
    # define q values
    q_values = np.linspace(0, 1, 10000)
    # loop through honest sizes
    for honest_size in honest_sizes:
        #initialize lists to store Qx values
        Qx_values = []
        # initialize list to store key rates
        keyrates = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # append Qx values to list
            Qx_values.append(network_obj.Qx)
            keyrate_inf_val = keyrate_inf(network_obj.Qx, network_obj.pstar)
            keyrates.append(keyrate_inf_val)
        print(f"Network: {honest_size}/{full_size}")
        # plot key rate as a function of Qx
        plt.plot(Qx_values, keyrates, label=f'Honest links: {honest_size}')
    
    plt.plot(np.linspace(0,.5,1000), BB84_keyrate(np.linspace(0,.5,1000)), color = 'black', label='BB84-A', linestyle='dotted', linewidth=2)
    #plt.grid(True)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlim(0,.2)
    plt.xlabel('Noise $Q_x$')
    plt.ylabel('Asymptotic Key-Rate')
    #plt.title(f'Asymptotic Key Rates for {full_size} Total Links')
    plt.legend()
    plt.show()

def plot_noise_tolerance_vs_honest_links(full_size, honest_sizes, N):
    # define q values
    q_values = x = np.linspace(0, 1, 1000)
    # initialize lists to store noise tolerances
    tolerances = []
    inf_tolerances = []
    #honest ratio list

    honest_ratios = []
    # loop through honest sizes
    for honest_size in honest_sizes:
        # initialize lists
        honest_ratios.append(honest_size/full_size)
        keyrates = []
        inf_keyrates = []
        Qx_values = []
        for q in q_values:
            # create network object
            network_obj = network(full_size, honest_size, depolarization(q))
            # calculate key rate
            keyrate_val = keyrate(N, network_obj.Qx, network_obj.pstar)
            inf_keyrate_val = keyrate_inf(network_obj.Qx, network_obj.pstar)
            Qx_values.append(network_obj.Qx)
            # append key rate to list
            keyrates.append(max(0,keyrate_val))
            inf_keyrates.append(max(0,inf_keyrate_val))
        # calculate noise tolerance
        tolerance = noise_tolerance(keyrates, Qx_values)
        inf_tolerance = noise_tolerance(inf_keyrates, Qx_values)
        # append noise tolerance to list
        tolerances.append(tolerance)
        inf_tolerances.append(inf_tolerance)
        print(f"Honest size: {honest_size}/{full_size}", "Noise tolerance:", tolerance, "Inf tolerance:", inf_tolerance)

    bb84_keyrates = []
    for noise in Qx_values:
        bb84_keyrates.append(max(0,finite_BB84_keyrate(N, noise)))
    finite_BB84_tolerance = noise_tolerance(bb84_keyrates, Qx_values)
    
    #splines
    finite_spline = PchipInterpolator(honest_ratios, tolerances)
    inf_spline = PchipInterpolator(honest_ratios, inf_tolerances)
    
    # plot noise tolerance as a function of honest links
    plt.plot(x, inf_spline(x), label='Asymptotic')
    plt.plot(x, finite_spline(x), label='Finite-Key')
    
    plt.axhline(y=0.11, color='black', linestyle='--', label='BB84-A')
    plt.axhline(y=finite_BB84_tolerance, color='black', linestyle='dotted', label='BB84-F')
    
    plt.plot(honest_ratios, tolerances, 'o')
    plt.plot(honest_ratios, inf_tolerances, 'o')
    
    plt.xlabel('Ratio of Honest Links')
    plt.ylabel('Noise Tolerance')
    plt.legend()
    plt.show()