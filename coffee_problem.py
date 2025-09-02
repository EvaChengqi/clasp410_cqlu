#!/usr/bin/env python3
'''
Solve the coffee problem to learn how to drink coffee effectively
'''

import numpy as np
import matplotlib.pyplot as plt

def solve_temp(t, T_init, T_env, k):
    '''
    This function returns temperature as a function of time 
    using Newton's law of cooling.

    parameters
    ----------
    t: Numpy array
        an array of time values in seconds
    T_init: floating point, defaults to 90
        Initial temperature in celsius 
    T_env: floating point, default to 20
        ambient air temperature in celsius 
    k: floating oiubtm default to 1/300
        heat transfer coefficient in 1/s

    ------------
    t_coffee: Numpy array
        Temperature corresponding to time t 

    '''

    t_coffee = T_env + (T_init - T_env) * np.exp(-k*t)

    return t_coffee
    
