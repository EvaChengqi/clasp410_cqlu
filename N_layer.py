#!/usr/bin/env python3

'''
This file solves the N-layer atmosphere problem for Lab 01 and all subparts

TO REPRODUCE THE VALES AND PLOTS IN MY REPORT, DO THIS:
<tbd>
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# physical constants
sigma=5.67e-8 #Stefan-Boltzmann constant, units: W/m2/K−4


# Question 1 and 2 
def n_layer_atoms(nlayers, epsilon =.1, albedo = 0.33, s0=1350, debug=False):
    '''
    Solve the N-layer gray atmosphere energy balance model for the Earth.
    ----------
    nlayers : int
        Number of atmospheric layers in the model (N ≥ 0).
        - 0 corresponds to no atmosphere (bare rock planet).
        - 1 or more layers introduce greenhouse trapping.

    epsilon : float, defaults to 0.1
        The emissivity/absorptivity of each atmospheric layer (0 ≤ ε ≤ 1).
        - ε = 1.0 means layers are perfect blackbodies in the infrared.
        - Smaller ε makes the layers more transparent.

    albedo : float, defaults to 0.33
        The surface albedo (reflectivity) of the planet.
        Fraction of incoming solar radiation reflected back to space.

    s0 : float, defaults to 1350
        Incoming solar constant (W/m^2).

    debug : bool, defaults to False
        If True, print out the coefficient matrix A and vector b for inspection.

    Returns
    ------------
    temps : numpy.ndarray of shape (nlayers+1,)
        Temperatures of the system in Kelvin (K).
        - Index 0 = surface temperature
        - Indices 1..N = atmospheric layer temperatures from bottom to top

    Notes
    ------------
    - The function constructs and solves a linear system A·x = b
    - Fluxes are converted back to temperatures via the Stefan-Boltzmann law.
    - Surface temperature uses ε=1, while atmospheric layers divide by ε
      to account for partial emissivity.
    '''
    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j:
                A[i,j] = -2 + 1*(j==0)
            else:
                A[i,j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j-i)-1)

    if debug: 
        print(A)

    b[0] = -.25 * s0 *(1-albedo) # What should go here?


    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!

    temps = np.zeros(nlayers + 1)
    # Emissity for earth =1
    temps[0] = np.power(fluxes[0] / sigma, 1/4)
    
    # Emissity for other atom layers = epsilon
    temps[1:] = np.power(fluxes[1:] / (sigma * epsilon), 1/4)

    return temps



# Question 3, experiments 1
#Graph of surface temperature versus emissivity
epsilon_values = np.linspace(0.01, 1, 100) 
surface_temps = [n_layer_atoms(nlayers=1, epsilon=e, s0=1350)[0] for e in epsilon_values]

# Finding the emissivity corresponding to the temperature closest to 288K
target_temp = 288
closest_temp_idx = np.abs(np.array(surface_temps) - target_temp).argmin()
predicted_epsilon = epsilon_values[closest_temp_idx]
predicted_temp = surface_temps[closest_temp_idx]

plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, surface_temps, color='blue', linewidth=2, label='Model Prediction')
plt.scatter([predicted_epsilon], [predicted_temp], color='red', s=100, zorder=5, label=f'Predicted Point ({predicted_epsilon:.2f}, {predicted_temp:.2f}K)')
plt.axhline(y=target_temp, color='gray', linestyle='--', linewidth=1, label=f'Target Temp ({target_temp}K)')
plt.axvline(x=predicted_epsilon, color='gray', linestyle='--', linewidth=1)
plt.text(0.1, 288, f'  Target: 288K', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=12)
plt.text(predicted_epsilon, 240, f'ε ≈ {predicted_epsilon:.2f}', verticalalignment='bottom', horizontalalignment='right', color='black', fontsize=12)


plt.xlabel('Emissivity (ε)')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs. Emissivity for a Single-Layer Atmosphere')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Experiments 1: \n")
print(f"For a surface temperature of 288K, the model predicts an atmospheric emissivity of approximately {predicted_epsilon:.2f}.")
print("\n")



# Question 3, experiments 2
num_layers = np.arange(1, 101, 1) 
surface_temps = [n_layer_atoms(nlayers=n, epsilon=0.255, s0=1350)[0] for n in num_layers]

plt.figure(figsize=(10, 6))
plt.plot(num_layers, surface_temps, color='green', marker='o', linestyle='-', linewidth=2)
plt.xlabel('Number of Layers')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature vs. Number of Layers (ε = 0.255)')
plt.tight_layout()
plt.grid(True)
plt.show()



# Question 4, planet Venus
num_layers = np.arange(1, 101, 1) 
surface_temps = [n_layer_atoms(nlayers=n, epsilon=1, albedo = 0.8, s0=2600)[0] for n in num_layers]

plt.figure(figsize=(10, 6))
plt.plot(num_layers, surface_temps, color='blue', marker='o', linestyle='-', linewidth=2)
plt.xlabel('Number of Layers')
plt.ylabel('Venus Surface Temperature (K)')
plt.title('Venus Surface Temperature vs. Number of Layers (ε = 1)')
plt.tight_layout()
plt.grid(True)
plt.show()




# Question 5 
def nuclear_winter_atoms(nlayers, epsilon = 0.5, albedo = 0.33, s0=1350, debug=False):
    '''
    Solve the N-layer model for a nuclear winter scenario.
    Solar flux is completely absorbed by the topmost layer.
    '''
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j:
                A[i,j] = -2 + 1*(j==0)
            else:
                A[i,j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j-i)-1)

    # Nuclear winter modification:
    # No solar flux reaches the surface (b[0] = 0)
    b[0] = 0.0
    # All solar flux is absorbed by the topmost layer (b[nlayers])
    b[nlayers] = -.25 * s0 *(1-albedo)

    if debug: 
        print(A)

    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)

    temps = np.zeros(nlayers + 1)
    
    temps[0] = np.power(fluxes[0] / sigma, 1/4)
    temps[1:] = np.power(fluxes[1:] / (sigma * epsilon), 1/4)

    return temps

# Run the nuclear winter simulation
# Using 5 layers, emissivity of 0.5, and solar constant s0 = 1350W/m2
temps_nuclear_winter = nuclear_winter_atoms(nlayers=5, epsilon=0.5, s0=1350, albedo=0.33)

# Print the surface temperature
print(f"The resulting surface temperature under a nuclear winter scenario is: {temps_nuclear_winter[0]:.2f} K")

# Plot the altitude-temperature profile
nu_num_layers = np.arange(1, 101, 1) 
nu_surface_temps = [n_layer_atoms(nlayers=n, epsilon=0.5, s0=1350)[0] for n in num_layers]

plt.figure(figsize=(10, 6))
plt.plot(nu_num_layers, nu_surface_temps, color='red', marker='o', linestyle='-', linewidth=2)
plt.xlabel('Number of Atmospheric Layers')
plt.ylabel('Nuclear Winter Earth Surface Temperature (K)')
plt.title('Nuclear Winter Temperature Profile (ε=0.5)')
plt.grid(True)
plt.tight_layout()
plt.show()
