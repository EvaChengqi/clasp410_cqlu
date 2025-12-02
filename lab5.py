#!/usr/bin/env python3

'''
Lab 5: Snowball Earth.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with `npoints` cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be `dLat/2` from 0 degrees and the
    last point will be `180 - dLat/2`.

    Parameters
    ----------
    npoints : int, defaults to 18
        Number of grid points to create.

    Returns
    -------
    dLat : float
        Grid spacing in latitude (degrees)
    lats : numpy array
        Locations of all grid cell centers.
    '''

    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''
    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.

    Returns
    --------
    lats : Numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        180 is north.
    Temp : Numpy array
        Temperature as a function of latitude.
    '''

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):

        # Update Albedo:
        loc_ice = Temp <= -10 # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd


        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp


def problem1():
    '''
    Create solution figure for Problem 1 (also validate our code qualitatively)
    '''

    # Get warm Earth initial condition.
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Create a fancy plot!
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, label='Initial Condition')
    ax.plot(lats-90, temp_diff, label='Diffusion Only')
    ax.plot(lats-90, temp_sphe, label='Diffusion + Spherical Corr.')
    ax.plot(lats-90, temp_alls, label='Diffusion + Spherical Corr. + Radiative')

    # Customize like those annoying insurance commercials
    ax.set_title('Solution after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')
    plt.show()


def test_functions():
    '''Test our functions'''

    print('Test gen_grid')
    print('For npoints=5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_grid(5)
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")


def problem2a():
    # 1. Prepare reference curve
    _, lats = gen_grid(18)
    ref_temp = temp_warm(lats)

    # 2. Parameter ranges
    lam_values = [0,15,50,100,150]   # 5 values
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 7 values

    # Storage
    RMSE_grid = np.zeros((len(eps_values), len(lam_values)))
    results = {}   # (eps, lam) -> temp curve

    best_rmse = float("inf")
    best_params = (None, None)

    # 3. Sweep parameters
    for i, eps in enumerate(eps_values):
        for j, lam in enumerate(lam_values):
            _, temp_final = snowball_earth(
                lam=lam, emiss=eps, tfinal=5000, apply_insol=True, apply_spherecorr=True
            )
            results[(eps, lam)] = temp_final

            rmse = np.sqrt(np.mean((temp_final - ref_temp) ** 2))
            RMSE_grid[i, j] = rmse

            print(f"eps={eps}, lam={lam}, rmse={rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (lam, eps)

    print("\nBest fit parameters:")
    print(f"λ = {best_params[0]}, ε = {best_params[1]}, RMSE = {best_rmse:.4f}")

    
    for i, eps in enumerate(eps_values):
        plt.figure(figsize=(8, 6))
        # Warm Earth reference line
        plt.plot(lats - 90, ref_temp, "k--", linewidth=2, label="Warm Earth (Reference)")

        for lam in lam_values:
            temp_curve = results[(eps, lam)]
            plt.plot(lats - 90, temp_curve, label=f"λ={lam}")

        plt.title(f"ε={eps} (Warm-Earth equilibrium)")
        plt.xlabel("Latitude (°)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return best_params


def problem2b():
    # 1. Prepare reference curve
    _, lats = gen_grid(18)
    ref_temp = temp_warm(lats)

    # 2. Parameter ranges
    lam_values = [50,55,60,65,70,75, 80,85,90,95,100]   # 5 values
    eps_values = [0.7]  # 7 values

    # Storage
    RMSE_grid = np.zeros((len(eps_values), len(lam_values)))
    results = {}   # (eps, lam) -> temp curve

    best_rmse = float("inf")
    best_params = (None, None)

    # 3. Sweep parameters
    for i, eps in enumerate(eps_values):
        for j, lam in enumerate(lam_values):
            _, temp_final = snowball_earth(
                lam=lam, emiss=eps, tfinal=5000, apply_insol=True, apply_spherecorr=True
            )
            results[(eps, lam)] = temp_final

            rmse = np.sqrt(np.mean((temp_final - ref_temp) ** 2))
            RMSE_grid[i, j] = rmse

            print(f"eps={eps}, lam={lam}, rmse={rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (lam, eps)

    print("\nBest fit parameters:")
    print(f"λ = {best_params[0]}, ε = {best_params[1]}, RMSE = {best_rmse:.4f}")

    return best_params


def problem3(lam_opt=55, eps_opt=0.7):
    dlat, lats = gen_grid(18)
    
    # Case 1: Hot Earth (60C everywhere)
    _, t_hot = snowball_earth(lam=lam_opt, emiss=eps_opt, init_cond=60.0,
                              albgnd = 0.3
                              , apply_insol=True, apply_spherecorr=True)
    
    # Case 2: Cold Earth (-60C everywhere)
    _, t_cold = snowball_earth(lam=lam_opt, emiss=eps_opt, init_cond=-60.0,
                               albgnd = 0.3
                               , apply_insol=True, apply_spherecorr=True)
    
    # Case 3: Flash Freeze
    # Start with Warm Earth curve, but force albedo to 0.6 initially.
    # To do this, we use the 'flash_freeze' flag implemented in the solver.
    _, t_flash = snowball_earth(lam=lam_opt, emiss=eps_opt, init_cond=temp_warm, 
                                albgnd = 0.6,
                                apply_insol=True, apply_spherecorr=True)

    print(f"Global Avg Temp (Hot Start):   {np.mean(t_hot):.2f} C")
    print(f"Global Avg Temp (Cold Start):  {np.mean(t_cold):.2f} C")
    print(f"Global Avg Temp (Flash Freeze):{np.mean(t_flash):.2f} C")

    plt.figure(figsize=(10, 6))
    plt.plot(lats-90, t_hot, 'r-', label='Start Hot (60C)')
    plt.plot(lats-90, t_cold, 'b-', label='Start Cold (-60C)')
    plt.plot(lats-90, t_flash, 'g--', label='Flash Freeze ')
    plt.plot(lats-90, temp_warm(lats), 'k:', label='Warm Earth Ref')
    plt.title(" Sensitivity to Initial Conditions")
    plt.xlabel("Latitude")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def problem4(lam_opt=55, eps_opt=0.7):

    gammas_up = np.arange(0.4, 1.45, 0.05)
    gammas_down = np.arange(1.4, 0.35, -0.05)
    
    avg_temps_up = []
    avg_temps_down = []
    
    current_temp = -60.0   # scalar initial condition (model will broadcast)

    for g in gammas_up:
        _, current_temp = snowball_earth(
            lam=lam_opt,
            emiss=eps_opt,
            solar = g * 1370,       
            init_cond=current_temp,
            tfinal=20000,
            apply_insol=True,
            apply_spherecorr=True
        )
        avg_temps_up.append(np.mean(current_temp))
    

    for g in gammas_down:
        _, current_temp = snowball_earth(
            lam=lam_opt,
            emiss=eps_opt,
            solar = g * 1370,     
            init_cond=current_temp,
            tfinal=20000,
            apply_insol=True,
            apply_spherecorr=True
        )
        avg_temps_down.append(np.mean(current_temp))


    plt.figure(figsize=(10, 6))
    plt.plot(gammas_up, avg_temps_up, 'r-o', label='Increasing Solar Flux')
    plt.plot(gammas_down, avg_temps_down, 'b-x', label='Decreasing Solar Flux')
    plt.axhline(y=-10, color='gray', linestyle='--', alpha=0.5, label='Freezing Threshold')

    plt.title("Impact of solar forcing on snowball earth")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Global Average Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_temp_warm():
    # Generate 18-point latitude grid
    dlat, lats = gen_grid(18)
    
    # Calculate warm Earth temperature distribution
    temp = temp_warm(lats)
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(lats - 90, temp, 'r-', label='Warm Earth (temp_warm)')
    plt.xlabel("Latitude (°)")
    plt.ylabel("Temperature (°C)")
    plt.title("Modern 'Warm Earth' Reference Temperature")
    plt.grid(True)
    plt.legend()
    plt.show()

