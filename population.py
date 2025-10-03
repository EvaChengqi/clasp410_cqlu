#!/usr/bin/env python3
'''
Lab 2: Population Control
Andrew Inda

This script models Lab 2, solves Lotka_Volterra equations
for competition and predator_prey systems. Both Euler and
RK8 solvers are used.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
plt.style.use("seaborn-v0_8")


## Derivative functions


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    """
    Competition model ODEs. Two species competing.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Logistic growth - competition terms
    dN1dt = a * N1 * (1 - N1) - b * N1 * N2
    dN2dt = c * N2 * (1 - N2) - d * N1 * N2
    return [dN1dt, dN2dt]


def dNdt_predprey(t, N, a=1, b=2, c=1, d=3):
    """
    Predator-prey model ODEs. Prey growth and predator hunting them.

    Inputs:
        t : time (float)
        N : [N1, N2] populations
        a, b, c, d : model parameters
    Outputs:
        [dN1dt, dN2dt]
    """
    # Unpack species
    N1, N2 = N
    # Prey grows, eaten by predator
    dN1dt = a * N1 - b * N1 * N2
    # Predator dies, grows when eating prey
    dN2dt = -c * N2 + d * N1 * N2
    return [dN1dt, dN2dt]


## Euler solver
def euler_solve(func, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    Euler solver (fixed step). Models the populations step by step.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Build time array
    time = np.arange(0, t_final + dT, dT)
    # Storage arrays
    N1 = np.zeros_like(time)
    N2 = np.zeros_like(time)
    # Set initial values
    N1[0], N2[0] = N1_init, N2_init

    # March forward in time
    for i in range(1, len(time)):
        dN1dt, dN2dt = func(time[i-1], [N1[i-1], N2[i-1]], a, b, c, d)
        # Euler update: new = old + step * slope
        N1[i] = N1[i-1] + dT * dN1dt
        N2[i] = N2[i-1] + dT * dN2dt
    return time, N1, N2


## RK8 solver


def solve_rk8(func, N1_init=0.3, N2_init=0.6, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    """
    RK8 solver (DOP853, adaptive step). Models smoother populations and takes smaller steps when necessary.

    Inputs:
        func : derivative function
        N1_init, N2_init : initial populations
        dT : max step size (years)
        t_final : total time (years)
        a, b, c, d : model parameters
    Outputs:
        time : array of times
        N1 : array of N1 values
        N2 : array of N2 values
    """
    # Call SciPy ODE solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=(a, b, c, d), method="DOP853", max_step=dT)
    return result.t, result.y[0], result.y[1]


## LAB QUESTIONS 

# # Question 1, reporduce 
a, b, c, d = 1, 2, 1, 3
t_final = 100

# step size = 1 year (Competition model)
time_euler_c, N1_euler_c, N2_euler_c = euler_solve(
    dNdt_comp, N1_init=0.3, N2_init=0.6, dT=1, t_final=t_final, a=a, b=b, c=c, d=d
)
time_rk8_c, N1_rk8_c, N2_rk8_c = solve_rk8(
    dNdt_comp, N1_init=0.3, N2_init=0.6, dT=10, t_final=t_final, a=a, b=b, c=c, d=d
)

# step size = 0.05 year (Predator-prey model)
time_euler_p, N1_euler_p, N2_euler_p = euler_solve(
    dNdt_predprey, N1_init=0.3, N2_init=0.6, dT=0.05, t_final=t_final, a=a, b=b, c=c, d=d
)
time_rk8_p, N1_rk8_p, N2_rk8_p = solve_rk8(
    dNdt_predprey, N1_init=0.3, N2_init=0.6, dT=10, t_final=t_final, a=a, b=b, c=c, d=d
)


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Competition model
ax[0].plot(time_euler_c, N1_euler_c, "b-", label="N1 Euler")
ax[0].plot(time_euler_c, N2_euler_c, "r-", label="N2 Euler")
ax[0].plot(time_rk8_c, N1_rk8_c, "b--", label="N1 RK8")
ax[0].plot(time_rk8_c, N2_rk8_c, "r--", label="N2 RK8")
ax[0].set_title("Lotka-Volterra Competition Model")
ax[0].set_xlabel("Time (years)")
ax[0].set_ylabel("Population/Carrying Cap")
ax[0].legend()

# Predator-prey model
ax[1].plot(time_euler_p, N1_euler_p, "r-", label="N1 (prey) Euler")
ax[1].plot(time_euler_p, N2_euler_p, "b-", label="N2 (predator) Euler")
ax[1].plot(time_rk8_p, N1_rk8_p, "r--", label="N1 (prey) RK8")
ax[1].plot(time_rk8_p, N2_rk8_p, "b--", label="N2 (predator) RK8")
ax[1].set_title("Lotka-Volterra Predator-Prey Model")
ax[1].set_xlabel("Time (years)")
ax[1].set_ylabel("Population/Carrying Cap")
ax[1].legend()

plt.tight_layout()
plt.show()


# # Question 1, test different step size
a, b, c, d = 1, 2, 1, 3
t_final = 100

# Euler solver with different step sizes for Competition model
time_euler_1, N1_euler_1, N2_euler_1 = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=1.0, t_final=t_final, a=a, b=b, c=c, d=d)
time_euler_05, N1_euler_05, N2_euler_05 = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=0.5, t_final=t_final, a=a, b=b, c=c, d=d)
time_euler_01, N1_euler_01, N2_euler_01 = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=0.1, t_final=t_final, a=a, b=b, c=c, d=d)

# RK8 solver
time_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init=0.3, N2_init=0.6, dT=10, t_final=t_final, a=a, b=b, c=c, d=d)

# Plot comparison
plt.figure(figsize=(10,5))
plt.plot(time_euler_1, N1_euler_1, "b-", label="N1 Euler dT=1.0")
plt.plot(time_euler_05, N1_euler_05, "g-", label="N1 Euler dT=0.5")
plt.plot(time_euler_01, N1_euler_01, "c-", label="N1 Euler dT=0.1")
plt.plot(time_rk8, N1_rk8, "b--", label="N1 RK8")

plt.plot(time_euler_1, N2_euler_1, "r-", label="N2 Euler dT=1.0")
plt.plot(time_euler_05, N2_euler_05, "m-", label="N2 Euler dT=0.5")
plt.plot(time_euler_01, N2_euler_01, "y-", label="N2 Euler dT=0.1")
plt.plot(time_rk8, N2_rk8, "r--", label="N2 RK8")

plt.xlabel("Time (years)")
plt.ylabel("Population / Carrying Capacity")
plt.title("Lotka-Volterra Competition Model: Euler vs RK8")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# Question 2: Uncomment to see all the graphs 
# uncomment this to get the first tryout 
initial_conditions = [(0.1,0.1), (0.3,0.6), (0.6,0.3)] 
coefficients = [(1,2,1,3), (1.5,1,1,2), (1,3,1,1)]

# uncomment this to get the figure reach to equilibrium
# initial_conditions = [(0.3,0.3)]
# coefficients = [(1,0.5,1,0.5), (1,0.8,1,0.8),(1,1,1,1)]

colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta'])

labels = [] 
for N1_init, N2_init in initial_conditions:
    for a,b,c,d in coefficients:
        t_rk, N1_r, N2_r = solve_rk8(dNdt_comp, N1_init, N2_init, dT=10, t_final=100, a=a,b=b,c=c,d=d)
        
        color1 = next(colors)
        color2 = next(colors)
        
        line1, = plt.plot(t_rk, N1_r, color=color1, alpha=0.7)
        line2, = plt.plot(t_rk, N2_r, color=color2, alpha=0.7)
        
        labels.append((line1, f"N1 init=({N1_init},{N2_init}) coef=({a},{b},{c},{d})"))
        labels.append((line2, f"N2 init=({N1_init},{N2_init}) coef=({a},{b},{c},{d})"))

plt.xlabel("Time (years)")
plt.ylabel("Population / Carrying Capacity")
plt.title("Lotka-Volterra Competition Model - Multiple Initial Conditions & Coefficients")
plt.tight_layout()

for i, (line, label) in enumerate(labels):
    plt.text(105, 0.9 - 0.04*i, label, color=line.get_color(), fontsize=8, va='top')

plt.xlim(0, 200)  
plt.show()


#Question 3 
# Figure 3: uncomment this to try different values of N2
# initial_conditions = [(0.5, 0.2),(0.5, 0.4),(0.5, 0.6)]
# coefficients = [(1, 0.5, 0.8, 0.4)]

# Figure 4: uncomment this to try different values of N1
# initial_conditions = [(0.3, 0.4),(0.5, 0.4),(0.7, 0.4)]
# coefficients = [(1, 0.5, 0.8, 0.4)]

# Figure 5: uncomment this to try different values of a
# initial_conditions = [(0.5, 0.2)]
# coefficients = [(1, 0.5, 0.8, 0.4), (1.2, 0.5, 0.8, 0.4), (1.3, 0.5, 0.8, 0.4)]

# Figure 6: uncomment this to try different values of c
# initial_conditions = [(0.5, 0.2)]
# coefficients = [(1, 0.3, 0.5, 0.4), (1, 0.5, 0.5, 0.4),(1, 0.8, 0.5, 0.4)]

# Figure 7: uncomment this to try different values of c
# initial_conditions = [(0.5, 0.2)]
# coefficients = [(1, 0.5, 0.5, 0.4), (1, 0.5, 0.7, 0.4),(1, 0.5, 0.9, 0.4)]

# Figure 8: uncomment this to try different values of d
# initial_conditions = [(0.5, 0.2)]
# coefficients = [(1, 0.3, 0.5, 0.4), (1, 0.3, 0.5, 0.6),(1, 0.3, 0.5, 0.8)]


# colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

# plt.figure(figsize=(12,6))
# for N1_init, N2_init in initial_conditions:
#     for a,b,c,d in coefficients:
#         t, N1, N2 = solve_rk8(dNdt_predprey, N1_init, N2_init, dT=0.01, t_final=100, a=a, b=b, c=c, d=d)
#         color1 = next(colors)
#         color2 = next(colors)
#         plt.plot(t, N1, color=color1, alpha=0.7,
#                  label=f"Prey init=({N1_init},{N2_init}) coef=({a},{b},{c},{d})")
#         plt.plot(t, N2, color=color2, alpha=0.7,
#                  label=f"Predator init=({N1_init},{N2_init}) coef=({a},{b},{c},{d})")

# plt.xlabel("Time (years)")
# plt.ylabel("Population")
# plt.title("Predator-Prey Dynamics (Time Series)")
# plt.legend(fontsize=8, loc='upper right')
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8,6))
# for N1_init, N2_init in initial_conditions:
#     for a,b,c,d in coefficients:
#         t, N1, N2 = solve_rk8(dNdt_predprey, N1_init, N2_init, dT=0.01, t_final=100, a=a, b=b, c=c, d=d)
#         color1 = next(colors)
#         plt.plot(N1, N2, color=color1, alpha=0.7,
#                  label=f"init=({N1_init},{N2_init}) coef=({a},{b},{c},{d})")

# plt.xlabel("Prey Population (N1)")
# plt.ylabel("Predator Population (N2)")
# plt.title("Predator-Prey Phase Diagram")
# plt.legend(fontsize=8)
# plt.tight_layout()
# plt.show()