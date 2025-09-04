#!/usr/bin/env python3
'''
Solve the coffee problem to learn how to drink coffee effectively
'''

import numpy as np
import matplotlib.pyplot as plt

def solve_temp(t, T_init=90., T_env=20.0, k=1/300.):
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

def time_to_temp(T_final, T_init=90., T_env=20.0, k=1/300.):
    '''
    given the target temperature, determine how long it get temp using Newton's law of cooling.
        parameters
    ----------
    t: Numpy array
        an array of time values in seconds
    T_init: floating point, defaults to 90
        Initial temperature in celsius 
    T_env: floating point, default to 20
        ambient air temperature in celsius 
    k: floating point default to 1/300
        heat transfer coefficient in 1/s
    T_final:  
        final temperature after cooling 

    returns
    ------------
    t_coffee: Numpy array
        Temperature corresponding to time t 
    '''

    t = (-1/k) * np.log((T_final - T_env) / (T_init - T_env))
    return t

def verify_code():
    '''
    verify that out implmentation is correct 
    '''
    t_real = 60.0 * 10.76
    k = np.log(95./110.)/-120.0
    t_code = time_to_temp(120., T_init=180., T_env=70., k=k)
    print("Target solution is", t_real)
    print("Numerical solution is", t_code)
    print("Difference solution is", t_real-t_code)

#solve the actual problem with the function declared above
#first, do it quantitatively to the screen
t_1 = time_to_temp(65.)                # add cream at T=65 to get to 60 
t_2 = time_to_temp(60., T_init= 85.)   # add cream immediately
t_c = time_to_temp(60.)                # control case: NO cream

print(f"TIME TO DRINKABLE COFFEE:\n\tControl case = {t_c:.3f}s\n\tAdd cream later = {t_1:.1f}\n\tAdd cream now = {t_2:.1f}")
print(f"\tControl case = {t_c:.3f}s")
print(f"\tAdd cream later = {t_1:.1f}\n\tAdd cream now = {t_2:.1f}")

# Create time serires of temperature for cooling coffee
t = np.arange(0, 600., 0.5)
temp1 = solve_temp(t)
temp2 = solve_temp(t, T_init=85)

# Create our figure and plot stuff
fig,ax = plt.subplots(1, 1)
ax.plot(t, temp1, label=f'Add Cream Later (T={t_1:.1f}s)')
ax.plot(t, temp2, label=f'Add Cream Now (T={t_2:.1f}s)')

ax.legend()
ax.set_xlabel('Time(s)')
ax.set_ylabel('Temperature (C)')
ax.set_title('When to add cream: Getting coffee cooled quickly')

