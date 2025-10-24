import numpy as np
import warnings
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

t_kanger = np.array([-19.7, -21.0, 10.7, 8.5, 3.1, -17.,
                     -6.0, -8.4, 2.3, 8.4,-12.0, -16.9])


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()


def solve_heat(xstop=1, tstop=0.2, dx=0.2, dt=0.02, c2=1, lowerbound=0,
               upperbound=0):
    '''
    A function for solving the heat equation.
    Apply Neumann boundary conditions such that dU/dx = 0.

    Parameters
    ----------
    xstop : float, optional
        The spatial domain length (maximum depth or distance) in meters.
        The grid points are defined from 0 to `xstop` with step size `dx`.
        Default is 1.

    tstop : float, optional
        The total simulation time. The temporal domain runs from 0 to `tstop`
        with a time step of `dt`. Units should be consistent with `c2`.
        Default is 0.2.

    dx : float, optional
        The spatial resolution, i.e., the distance between adjacent grid points.
        Smaller values improve spatial accuracy but increase computation time.
        Default is 0.2.

    dt : float, optional
        The time step increment used in the simulation.
        Must satisfy the numerical stability condition (Courant criterion):
            c² * dt / dx² ≤ 0.5
        for stable solutions. Default is 0.02.

    c2 : float
        c^2, the square of the diffusion coefficient.

        lowerbound : None, scalar, or callable, optional
        The lower boundary condition (at x = 0).  
        - If `None`, a Neumann boundary (zero gradient) is applied.  
        - If a scalar is given, it enforces a constant Dirichlet boundary condition.  
        - If a function is given, it should accept time `t` and return temperature.  
        Default is 0.

    upperbound : None, scalar, or callable, optional
        The upper boundary condition (at x = xstop).  
        Same format as `lowerbound`.  
        Default is 0.

    Parameters
    ----------
    initial : func
        A function of position; sets the intial conditions at t=`trange[0]`
        Must accept an array of positions and return temperature at those
        positions as an equally sized array.
    upperbound, lowerbound : None, scalar, or func
        Set the lower and upper boundary conditions. If either is set to
        None, then Neumann boundary condtions are used and the boundary value
        is set to be equal to its neighbor, producing zero gradient.
        Otherwise, Dirichlet conditions are used and either a scalar constant
        is provided or a function should be provided that accepts time and
        returns a value.

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    '''

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
        warnings.warn(f'Stability criterion is not met: dt={dt} > dt_max={dt_max}. Solution will be unstable.')

    # Get grid sizes (plus one to include "0" as well.)
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions (Hardcoded to 0°C)
    U = np.zeros([M, N])
    # U[:, 0] = 4*x - 4*x**2

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

        # Apply boundary conditions:
        # Lower boundary (x=0 in your solver's index U[0,:]) -> Corresponds to bottom boundary
        if lowerbound is None:  # Neumann
            U[0, j+1] = U[1, j+1]
        elif callable(lowerbound):  # Time-dependent Dirichlet
            U[0, j+1] = lowerbound(t[j+1])
        else:
            U[0, j+1] = lowerbound

        # Upper boundary (x=xstop in your solver's index U[-1,:]) -> Corresponds to surface boundary
        if upperbound is None:  # Neumann
            U[-1, j+1] = U[-2, j+1]
        elif callable(upperbound):  # Time-dependent Dirichlet
            U[-1, j+1] = upperbound(t[j+1])
        else:
            U[-1, j+1] = upperbound

    # Return our pretty solution to the caller:
    return t, x, U


def heatmap_figure1(year=120):
    xstop = 100.0 
    tstop = year * 365 * 24 * 3600   
    dx = 1.0     
    c2 = 2.5e-7 # 7.889 m^2/year

    dt_max = dx**2 / (2 * c2)
    dt = dt_max / 1.1 
    lowerbound_func = lambda t: temp_kanger(t/ 86400) 
    
    # Run Solver
    time, x, heat = solve_heat(xstop=xstop, tstop=tstop, dx=dx, dt=dt,
                                c2=c2, lowerbound=lowerbound_func, 
                                upperbound=5)

    # Figure 1
    fig, axes = plt.subplots(1, 1)
    axes.invert_yaxis()
    map = axes.pcolor(time / (365*24*3600), x, heat, cmap='seismic', vmin=-50, vmax=50)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')

    axes.set_title("Ground Temperature: Kangerlussuaq, Greenland")
    axes.set_xlabel("Time (Years)")
    axes.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.show()


def Seasonal_figure_2(year=120):
    xstop = 100.0 
    tstop = year * 365 * 24 * 3600   # Simulate 300 years
    dx = 1.0     
    c2 = 2.5e-7  # Thermal diffusivity (m²/s)

    # Ensure numerical stability (Courant condition)
    dt_max = dx**2 / (2 * c2)
    dt = dt_max / 1.1 

    # Surface boundary (Kangerlussuaq temperature pattern)
    lowerbound_func = lambda t: temp_kanger(t / 86400) 
    
    # Run heat conduction solver
    time, x, U = solve_heat(xstop=xstop, tstop=tstop, dx=dx, dt=dt,
                            c2=c2, lowerbound=lowerbound_func, 
                            upperbound=5)

    # Check for steady state in the isothermal zone (deep region)
    isothermal_idx_start = int(60 / dx)  # Start checking at 60m depth
    loc_last_year = int(-365 * 24 * 3600 / dt)        # Last year index
    loc_prev_year = int(-2 * 365 * 24 * 3600 / dt)    # Year before that

    # Compute mean temperature for last two years in deep region
    temp_last_year = U[isothermal_idx_start:, loc_last_year:].mean()
    temp_prev_year = U[isothermal_idx_start:, loc_prev_year:loc_last_year].mean()
    temp_change = abs(temp_last_year - temp_prev_year)
    is_steady = temp_change < 0.01  # If temperature change < 0.01°C → steady

    #  Extract last year's winter & summer temperature profiles
    loc = int(-365 / (dt / 86400))  # Number of time steps in one year
    winter = U[:, loc:].min(axis=1)  # Minimum temperature (winter)
    summer = U[:, loc:].max(axis=1)  # Maximum temperature (summer)

    # Find active layer depth (where summer temp crosses 0°C)
    idx = np.where((summer[:-1] > 0.0) & (summer[1:] <= 0.0))[0]
    if idx.size:
        i = idx[0]
        x0, x1 = x[i], x[i+1]
        y0, y1 = summer[i], summer[i+1]
        # Linear interpolation to find exact depth where temp = 0°C
        active_layer_depth = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    else:
        active_layer_depth = xstop


    # Find permafrost bottom depth (where winter temp crosses 0°C)
    idxb = np.where((winter[:-1] <= 0.0) & (winter[1:] > 0.0))[0]
    if idxb.size:
        j = idxb[0]
        xb0, xb1 = x[j], x[j+1]
        yw0, yw1 = winter[j], winter[j+1]
        permafrost_bottom = xb0 + (0.0 - yw0) * (xb1 - xb0) / (yw1 - yw0)
    else:
        permafrost_bottom = xstop

    print(f"Active layer depth ≈ {active_layer_depth:.2f} m")
    print(f"Permafrost bottom depth ≈ {permafrost_bottom:.2f} m")

    # Plot seasonal temperature profiles
    fig, ax2 = plt.subplots(1, 1, figsize=(10,8))
    ax2.axvline(0, color='k', linestyle=':', label='0°C line')  # Reference line
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, label='Summer', linestyle='--')

    # Mark active layer and permafrost boundaries
    ax2.axhline(active_layer_depth, color='orange', linestyle='--', 
                label=f'Active Layer ~ {active_layer_depth:.1f} m')
    ax2.axhline(permafrost_bottom, color='purple', linestyle='--', 
                label=f'Permafrost Bottom ~ {permafrost_bottom:.1f} m')

    # Add surface temperature values for x=0 (optional visualization)
    winter_surface = winter[0]
    summer_surface = summer[0]
    ax2.text(0.2, 2, f"Winter surface: {winter_surface:.1f}°C", color='blue')
    ax2.text(0.2, 5, f"Summer surface: {summer_surface:.1f}°C", color='red')

    ax2.invert_yaxis()
    ax2.set_title("Ground Temperature Profile: Kangerlussuaq")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Depth (m)")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def run_permafrost_sim(temp_shift=0):
    """
    Solve the ground temperature profile for Kangerlussuaq under a given temperature shift.

    Parameters
    ----------
    temp_shift : float
        Uniform temperature increase (°C) added to the Kangerlussuaq climate curve
        to simulate global warming.

    Returns
    -------
    time : array
        Time points in seconds
    x : array
        Depth points in meters
    heat : 2D array
        Temperature matrix (depth x time)
    active_layer_depth : float
        Depth (m) of the active layer in summer
    permafrost_bottom : float
        Depth (m) of the permafrost bottom in winter
    permafrost_thickness : float
        Thickness (m) of the permafrost layer
    """

    # ----- Set spatial and temporal parameters -----
    xstop = 100.0                 # maximum depth (m)
    tstop = 120 * 365 * 24 * 3600 # simulate 300 years in seconds
    dx = 1.0                       # spatial resolution (m)
    c2 = 2.5e-7                    # thermal diffusivity squared (m^2/s)
    dt = (dx**2 / (2*c2)) / 1.1    # stable time step using CFL criterion

    # ----- Define upper boundary: Kangerlussuaq surface temperature + shift -----
    lowerbound_func = lambda t: temp_kanger(t / 86400) + temp_shift

    # ----- Solve the heat equation -----
    time, x, heat = solve_heat(
        xstop=xstop,
        tstop=tstop,
        dx=dx,
        dt=dt,
        c2=c2,
        lowerbound=lowerbound_func,
        upperbound=5    # lower boundary at 5°C (geothermal)
    )

    # ----- Extract winter and summer temperatures from last year -----
    loc = int(-365 / (dt / 86400))   # index for last year
    winter = heat[:, loc:].min(axis=1)
    summer = heat[:, loc:].max(axis=1)

    # ----- Calculate active layer depth (where summer temp crosses 0°C) -----
    idx = np.where((summer[:-1] > 0) & (summer[1:] <= 0))[0]
    if idx.size:
        i = idx[0]
        x0, x1 = x[i], x[i+1]
        y0, y1 = summer[i], summer[i+1]
        # linear interpolation to find exact depth at 0°C
        active_layer_depth = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    else:
        active_layer_depth = xstop

    # ----- Calculate permafrost bottom (where winter temp crosses 0°C) -----
    idxb = np.where((winter[:-1] <= 0) & (winter[1:] > 0))[0]
    if idxb.size:
        j = idxb[0]
        xb0, xb1 = x[j], x[j+1]
        yw0, yw1 = winter[j], winter[j+1]
        # linear interpolation to find exact depth at 0°C
        permafrost_bottom = xb0 + (0 - yw0) * (xb1 - xb0) / (yw1 - yw0)
    else:
        permafrost_bottom = xstop

    # ----- Calculate permafrost thickness -----
    permafrost_thickness = permafrost_bottom - active_layer_depth

    return time, x, heat, active_layer_depth, permafrost_bottom, permafrost_thickness


# ----- Loop over different warming scenarios -----
shifts = [0.5, 1, 3]  # °C temperature increases
for s in shifts:
    t, x, heat, active, pf_bottom, pf_thick = run_permafrost_sim(temp_shift=s)
    print(f"Temperature shift {s}°C: Active layer ≈ {active:.2f} m, "
          f"Permafrost thickness ≈ {pf_thick:.2f} m")
