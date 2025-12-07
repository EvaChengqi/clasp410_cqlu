
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use("seaborn-v0_8")

# Horizontal distance: km (1 grid cell = 1 km)
# Iceberg height: meters (m)
# Time step: years
# Melt rates: meters per year (m/year)


def initialize_iceberg(nx=200, init_width=25, min_h=98.0, max_h=164.0, source_idx=0):
    '''
    nx: number of columns (1 column = 1 km)
    init_width: Initial iceberg width in km
    min_h: Minimum iceberg height in meters (m)
    max_h: Maximum iceberg height in meters (m)
    source_idx: Source position in km (grid index)
    '''
    iceberg = np.zeros(nx)
    
    # Iceberg starts from the source index (0)
    start = source_idx
    end = source_idx + init_width 

    # random heights for initial iceberg columns
    iceberg[start:end] = np.random.uniform(min_h, max_h, init_width)
    return iceberg


def update_iceberg(h, u=1, min_h=98.0, max_h=164.0, source_idx=0, l_default=0.5):
    '''
    h: Iceberg height array [m]
    u: Drift speed in grid cells per year (≈ km/year)
    min_h: Minimum injected block height [m]
    max_h: Maximum injected block height [m]
    source_idx: Source index [km]
    l_default: Melt per time step [m/year]
    '''
    h[u:] = h[:-u]
    
    new_block_height = np.random.uniform(min_h, max_h)
    h[source_idx] = new_block_height

    if callable(l_default):
        loss = l_default(h)
    else:
        loss = l_default
    
    h = h - loss
    h = np.maximum(h, 0) 
    return h


def get_loss(h_array, t, base_O_ENV, L_BULK = 0.005, L_ENV  = 0.01, O_BULK = 0.03, O_ENV  = 0.2, ocean_idx=100, nx=200, mode = 'simple'):
    '''
    Calculates the mass loss rate (melting/erosion) for each iceberg column,
    based on a hybrid model incorporating both bulk melt (height-dependent)
    and environmental erosion (constant, location-dependent).

    The loss rate is differentiated based on whether the column is in the 
    Land (glacier/fjord) region or the Ocean (open sea) region.

    Parameters
    ----------
    h_array : numpy.ndarray
        The current height (mass) array of the iceberg columns. This is the 
        only required positional argument.

    L_BULK : float, optional
        Coefficient for Bulk Melt in the Land region.
        The bulk melt rate is calculated as `L_BULK * h`.
        Represents basal/submerged melting in relatively still, cold water.
        Default is 0.005.

    L_ENV : float, optional
        Coefficient for Environmental Erosion in the Land region.
        Represents constant mass loss per step due to minor surface effects
        (e.g., runoff, air temperature).
        Default is 0.01.

    O_BULK : float, optional
        Coefficient for Bulk Melt in the Ocean region.
        The bulk melt rate is calculated as `O_BULK * h`.
        Represents high basal/submerged melting due to warmer, circulating ocean currents.
        Default is 0.03.

    O_ENV : float, optional
        Coefficient for Environmental Erosion in the Ocean region.
        Represents constant mass loss per step due to major surface effects
        (e.g., strong wave erosion, high-speed turbulence at the waterline).
        Default is 0.2.

    ocean_idx : int, optional
        The array index that defines the boundary between the Land region (0 to `ocean_idx`-1) 
        and the Ocean region (`ocean_idx` to `nx`-1).
        Default is 100.

    nx : int, optional
        The total number of columns in the spatial domain (size of `h_array`).
        Default is 200.

    Returns
    -------
    loss_array : numpy.ndarray
        A 1D array of the same size as `h_array`, where each element represents 
        the total mass loss (height reduction) that occurs at that position 
        during the current time step.
    '''
    if mode == 'simple':
        loss_array = np.zeros_like(h_array)
        land_slice = slice(0, ocean_idx)

        # Calculate Land Loss: Bulk (h-dependent) + Environmental (constant)
        land_bulk_loss = L_BULK * h_array[land_slice]
        land_env_loss = L_ENV

        loss_array[land_slice] = land_bulk_loss + land_env_loss

        ocean_slice = slice(ocean_idx, nx)
        # Calculate Ocean Loss: Bulk (h-dependent) + Environmental (constant)
        ocean_bulk_loss = O_BULK * h_array[ocean_slice]
        ocean_env_loss = O_ENV
        loss_array[ocean_slice] = ocean_bulk_loss + ocean_env_loss

    loss_array[h_array < 1e-6] = 0
    return loss_array



def animate_iceberg(nx=200, nt=500, dt=0.5, max_h=164.0, ocean_start_idx=100):
    
    h = initialize_iceberg()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.axvspan(0, ocean_start_idx, color='lightgreen', alpha=0.3, label='Land (Source)')
    ax.axvspan(ocean_start_idx, nx, color='lightblue', alpha=0.3, label='Ocean')
    
    bar_container = ax.bar(np.arange(nx), h, width=1.0, color="skyblue")

    ax.set_ylim(0, max_h + 5)
    ax.set_title("Iceberg Growth at h[0] and Rightward Drift (Inline Params)")
    ax.set_xlabel(f"Position (Ocean starts at index {ocean_start_idx})")
    ax.set_ylabel("Height")
    ax.legend(loc='upper right')

    def animate(frame):
        nonlocal h
        h = update_iceberg(h, l_default=lambda arr: get_loss(arr, t=frame, base_O_ENV=0.1, mode='simple'))

        # update each bar’s height
        for bar, new_height in zip(bar_container, h):
            bar.set_height(new_height)

        ax.set_title(f"Iceberg Evolution  (t = {frame*dt:.1f} s)")
        return bar_container

    ani = animation.FuncAnimation(
        fig, animate, frames=nt, interval=50, blit=False, repeat=False
    )
    plt.tight_layout()
    plt.show()



# melting with temperate increase due to green house gas
def simulate_extent(nt=500, mode='simple'):
    """
    Run the iceberg update loop (without animation) and track the iceberg extent
    defined as the rightmost index where h > 0.
    """
    h = initialize_iceberg()
    extents = []

    for t in range(nt):
        h = update_iceberg(h, l_default=lambda arr: get_loss(arr, t=t, base_O_ENV=0.1, mode=mode))

        # iceberg extent = rightmost column with height > 0
        nonzero = np.where(h > 1e-6)[0]
        if len(nonzero) == 0:
            extent = 0
        else:
            extent = nonzero[-1]
        extents.append(extent)
    return extents



