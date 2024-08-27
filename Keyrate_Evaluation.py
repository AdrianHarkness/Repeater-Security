#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 11 12:11:19 2023

@author: adrianharkness
"""

from Network_Optimization import w_tilde
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    delta = np.sqrt(((N+2)*np.log(2/(epsilon**2))) / (m*(N)))
    #ideal-ideal state delta
    delta_prime = np.sqrt(np.log(2/epsilon) / (2*m))
   
    #keyrate evaluation
    #Do I need hamming weight of Qx? Qx isn't a binary string here...
    if pstar == 0.5:
        keyrate = (-1.2*(bin_entropy(Qx))) / N
    else:
        Q = ((Qx - pstar + delta_prime)/(1-2*pstar))+delta #66
        leak_ec = 1.2*(bin_entropy(Qx + delta))
        keyrate = (n/N)*(1 - bin_entropy(Q) - leak_ec) - (np.log2(1/epsilon))/N #66
        
    #bound keyrates to be between 0 and 1
    #keyrate = max(0, keyrate)
    if keyrate > 1:
        keyrate = 1
    return keyrate

#asymptotic key rate (eq 67)
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

#==============================================================================
#===================General Network Keyrate Calculations=======================

def general_network_keyrate(Qx, N, p_array, m_array, epsilon=1e-36):
    #number of signals in measured test set
    m = np.round(.07*N)
    #number of signals remaining for secret key
    n = N-m
    #ideal state delta
    delta = np.sqrt(((N+2)*np.log(2/(epsilon**2))) / (m*(N)))
    #ideal-ideal state delta
    delta_prime = np.sqrt(np.log(2/epsilon) / (2*m))

    #Maximize adversarial error
    w = w_tilde(Qx, delta_prime, p_array, m_array) #71
    #print(w)
    Q = w + delta
    #keyrate evaluation
    leak_ec = 1.2*(bin_entropy(Qx + delta))
    keyrate = (n/N)*(1 - bin_entropy(Q) - leak_ec) - (np.log2(1/epsilon))/N #70
        
    #bound keyrates to be between 0 and 1
    #keyrate = max(0, keyrate)
    if keyrate > 1:
        keyrate = 1
    return keyrate



#==============================================================================
#=======================BB84 Keyrate Calculations===============================

#Don't use
def BB84_F_2(N, Q, epsilon=1e-36):
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

#Use this instead
def BB84_F(N, Qx, epsilon=1e-36):
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
    nu = np.sqrt(((N)*(m+1)*np.log(2/epsilon)) / ((m**2)*n))
    #keyrate
    #keyrate = (n/N)*(np.log2(d) - 2.2*bin_entropy(Q + nu) - (Q + nu)*np.log2(d - 1))
    keyrate = (n/N)*(1 - bin_entropy(Qx + nu) - 1.2*bin_entropy(Qx + nu))

    return keyrate

#asymptotic BB84 keyrate
def BB84_A(noise):
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