#!/usr/bin/env python3

'''
A module for burning forests and making pestilence.
What a happy coding time.
'''

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set plot style:
plt.style.use('fivethirtyeight')

# Generate our custom segmented color map for this project.
# We can specify colors by names and then create a colormap that only uses
# those names. We have 3 funadmental states, so we want only 3 colors.
# Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
colors = ['tan', 'forestgreen', 'crimson']
forest_cmap = ListedColormap(colors)


def forest_fire(isize=3, jsize=3, nstep=4, pspread=1.0, pignite=0.0, pbare=0):
    '''
    Create a forest fire.

    Parameters
    ----------
    isize, jsize : int, defaults to 3
        Set size of forest in x and y direction, respectively.
    nstep : int, defaults to 4
        Set number of steps to advance solution.
    pspread : float, defaults to 1.0
        Set chance that fire can spread in any direction, from 0 to 1
        (i.e., 0% to 100% chance of spread.)
    pignite : float, defaults to 0.0
        Set the chance that a point starts the simulation on fire (or infected)
        from 0 to 1 (0% to 100%).
    pbare : float, defaults to 0.0
        Set the chance that a point starts the simulation on bare (or
        immune) from 0 to 1 (0% to 100%).
    '''

    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Set initial conditions for BURNING/INFECTED and BARE/IMMUNE
    # Start with BURNING/INFECTED:
    if pignite > 0:  # Scatter fire randomly:
        loc_ignite = np.zeros((isize, jsize), dtype=bool)
        while loc_ignite.sum() == 0:
            loc_ignite = rand(isize, jsize) <= pignite
        print(f"Starting with {loc_ignite.sum()} points on fire or infected.")
        forest[0, loc_ignite] = 3
    else:
        # Set initial fire to center:
        forest[0, isize//2, jsize//2] = 3

    # Set bare land/immune people:
    loc_bare = rand(isize, jsize) <= pbare
    forest[0, loc_bare] = 1

    # Loop through time to advance our fire.
    for k in range(nstep-1):
        # Assume the next time step is the same as the current:
        forest[k+1, :, :] = forest[k, :, :]
        # Search every spot that is on fire and spread fire as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire?
                if forest[k, i, j] != 3:
                    continue
                # Ah! it burns. Spread fire in each direction.
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (i > 0) and (forest[k, i-1, j] == 2):
                    forest[k+1, i-1, j] = 3
                # Spread "Down" (i to i+1)
                if (pspread > rand()) and (i < isize - 1) and (forest[k, i+1, j] == 2):
                    forest[k+1, i+1, j] = 3
                # Spread "East" (j to j-1)
                if (pspread > rand()) and (j > 0) and (forest[k, i, j-1] == 2):
                    forest[k+1, i, j-1] = 3
                # Spread "West" (j to j+1)
                if (pspread > rand()) and (j < jsize - 1) and (forest[k, i, j+1] == 2):
                    forest[k+1, i, j+1] = 3

                # Change buring to burnt:
                forest[k+1, i, j] = 1

    return forest


def plot_progression(forest):
    '''Calculate the time dynamics of a forest fire and plot them.'''

    # Get total number of points:
    ksize, isize, jsize = forest.shape
    npoints = isize * jsize

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc = forest == 2
    forested = 100 * loc.sum(axis=(1, 2))/npoints

    loc = forest == 1
    bare = 100 * loc.sum(axis=(1, 2))/npoints
    plt.plot(forested, label='Forested' )
    plt.plot(bare, label='Bare/Burnt')

    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Percent Total Forest')


def plot_forest2d(forest_in, itime=0):
    '''
    Given a forest of size (ntime, nx, ny), plot the itime-th moment as a
    2d pcolor plot.
    '''

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.subplots_adjust(left=.117, right=.974, top=.929, bottom=0.03)

    # Add our pcolor plot, save the resulting mappable object.
    map = ax.pcolor(forest_in[itime, :, :], vmin=1, vmax=3, cmap=forest_cmap)

    # Add a colorbar by handing our mappable to the colorbar function.
    cbar = plt.colorbar(map, ax=ax, shrink=.8, fraction=.08,
                        location='bottom', orientation='horizontal')
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['Bare/Burnt', 'Forested', 'Burning'])

    # Flip y-axis (corresponding to the matrix's X direction)
    # And label stuff.
    ax.invert_yaxis()
    ax.set_xlabel('Eastward ($km$) $\\longrightarrow$')
    ax.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax.set_title(f'The Seven Acre Wood at T={itime:03d}')

    # Return figure object to caller:
    return fig


def make_all_2dplots(forest_in, folder='results/'):
    '''
    For every time frame in `forest_in`, create a 2D plot and save the image
    in folder.
    '''

    import os

    # Check to see if folder exists, if not, make it!
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Make a buncha plots.
    ntime, nx, ny = forest_in.shape
    for i in range(ntime):
        print(f"\tWorking on plot #{i:04d}")
        fig = plot_forest2d(forest_in, itime=i)
        fig.savefig(f"{folder}/forest_i{i:04d}.png")
        plt.close('all')


def figure1():
    forest = forest_fire(isize=3, jsize=3, nstep=3, pspread=1.0, pignite=0.0, pbare=0.0)
    for t in range(forest.shape[0]):
        fig = plot_forest2d(forest, itime=t)
        plt.show()
    
    plot_progression(forest)
    plt.legend()
    plt.title('Forest Fire Progression (3x3 Grid)')
    plt.tight_layout()
    plt.show()


def figure2():
    forest_wide = forest_fire(isize=6, jsize=11, nstep=4, pspread=1.0, pignite=0.0, pbare=0.0)

    for t in range(forest_wide.shape[0]):
        fig = plot_forest2d(forest_wide, itime=t)
        plt.show()
    
    plot_progression(forest_wide)
    plt.legend()
    plt.title('Forest Fire Progression (6x11 Grid)')
    plt.tight_layout()
    plt.show()


def pspread_stimulate():
    isize, jsize, nstep = 30, 30, 50
    pignite, pbare = 0.02, 0.0
    pspread_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for pspread in pspread_values:
        forest = forest_fire(isize, jsize, nstep, pspread, pignite, pbare)


        ksize, isize, jsize = forest.shape
        npoints = isize * jsize
        forested = 100 * (forest == 2).sum(axis=(1, 2)) / npoints

        stabilized = False
        stable_step = None
        for t in range(1, len(forested)):
            if abs(forested[t] - forested[t - 1]) < 0.01:
                stabilized = True
                stable_step = t
                break


        print(f"{pspread:7.2f} | {forested[-1]:17.2f}% | ")
        
        plt.figure(figsize=(8, 5))   
        plot_progression(forest) 

        plt.title(f'Wildfire Evolution (Pspread={pspread})')
        plt.legend()
        plt.tight_layout()
        plt.show()                 


def pbare_stimulate():
    isize, jsize, nstep = 30, 30, 50
    pignite = 0.02
    pbare_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pspread = 0.6  

    for pbare in pbare_values:
        forest = forest_fire(isize, jsize, nstep, pspread, pignite, pbare)

        ksize, isize, jsize = forest.shape
        npoints = isize * jsize
        forested = 100 * (forest == 2).sum(axis=(1, 2)) / npoints
        stabilized = False
        stable_step = None
        for t in range(1, len(forested)):
            diff = abs(forested[t] - forested[t - 1])
            if diff < 0.01:
                stabilized = True
                stable_step = t
                break  

        print(f"{pbare:5.2f} | {forested[-1]:17.2f}% | ")

    
        plt.figure(figsize=(8, 5))  
        plot_progression(forest)     

        plt.title(f'Wildfire Evolution (Pbare={pbare}, Pspread={pspread})')
        plt.legend()
        plt.xlabel('Time (arbitrary units)')
        plt.ylabel('Percent Total Forest')
        plt.tight_layout()
        plt.show()                  


def disease_spread(isize=3, jsize=3, nstep=4, pspread=1.0, pignite=0.02, pbare=0, psurvival=0.0):
    """
    Simulate disease spread (Buckeyeitis) on a grid using a wildfire model.

    Status codes:
    2 -> Healthy
    3 -> Infected
    1 -> Recovered / Immune
    0 -> Dead

    Parameters
    ----------
    isize, jsize : int
        Grid dimensions
    nstep : int
        Number of time steps
    pspread : float
        Probability that disease spreads to a neighboring cell
    pignite : float
        Probability that a healthy cell is initially infected
    pbare : float
        Probability that a cell is initially immune (vaccinated)
    psurvival : float
        Probability that an infected person dies
    """

    # Initialize grid: all healthy
    grid = np.full((nstep, isize, jsize), 2)

    # Set initial immune population (vaccinated)
    loc_immune = rand(isize, jsize) < pbare
    grid[0, loc_immune] = 1

    # Initialize infections
    loc_infected = np.zeros((isize, jsize), dtype=bool)
    while loc_infected.sum() == 0:
        loc_infected = rand(isize, jsize) < pignite
    grid[0, loc_infected] = 3

    # Time evolution
    for t in range(nstep-1):
        grid[t+1, :, :] = grid[t, :, :]
        for i in range(isize):
            for j in range(jsize):
                if grid[t, i, j] != 3:
                    continue  # Only spread from infected cells

                # Spread disease to neighbors if healthy
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < isize and 0 <= nj < jsize and grid[t, ni, nj] == 2:
                        if rand() < pspread:
                            grid[t+1, ni, nj] = 3

                # Determine outcome of current infected cell
                if rand() < psurvival:
                    grid[t+1, i, j] = 0  # Person dies
                else:
                    grid[t+1, i, j] = 1  # Person recovers and becomes immune

    return grid


def plot_disease_progression(grid):
    nsteps, isize, jsize = grid.shape
    total = isize * jsize

    healthy = 100 * (grid == 2).sum(axis=(1,2)) / total
    infected = 100 * (grid == 3).sum(axis=(1,2)) / total
    recovered = 100 * (grid == 1).sum(axis=(1,2)) / total
    dead = 100 * (grid == 0).sum(axis=(1,2)) / total

    plt.plot(healthy, label='Healthy')
    plt.plot(infected, label='Infected')
    plt.plot(recovered, label='Recovered/Immune')
    plt.plot(dead, label='Dead')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Percentage of population')
    plt.legend()


def vaccine():
    isize, jsize = 30, 30
    nstep = 50
    pspread = 0.6  
    pignite = 0.02  

    psurvival = 0.3
    pbare_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


    for pbare in pbare_values:
        grid = disease_spread(isize, jsize, nstep,
                            pspread=pspread,
                            pignite=pignite,
                            pbare=pbare,
                            psurvival=psurvival)
            
        plt.figure(figsize=(8,5))
        plot_disease_progression(grid)
        plt.title(f'Disease Evolution: Psurvival={psurvival}, Pbare={pbare}')
        plt.tight_layout()
        plt.show()


def psurvival():
    """
    Explore how different disease mortality rates (psurvival) affect disease spread dynamics.
    """
    # Simulation parameters
    isize, jsize = 30, 30
    nstep = 50
    pspread = 0.6       # fixed disease spread probability
    pignite = 0.02      # initial infection probability
    pbare = 0.2         # early vaccination rate (fixed for this experiment)
    
    # Different mortality rates to explore (0 = all survive, 1 = all die)
    psurvival_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for psurvival in psurvival_values:
        # Run simulation
        grid = disease_spread(
            isize, jsize, nstep,
            pspread=pspread,
            pignite=pignite,
            pbare=pbare,
            psurvival=psurvival
        )

        # Plot progression
        plt.figure(figsize=(8, 5))
        plot_disease_progression(grid)
        plt.title(f'Disease Evolution: Early Vaccine Rate Pbare={pbare}, Mortality Rate Psurvival={psurvival}')
        plt.tight_layout()
        plt.show()
