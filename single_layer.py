'''
solve global warming with 1-layer atmosphere energy balance 
'''
import numpy as np
import matplotlib.pyplot as plt

# Change our matplotlib style
plt.style.use('fivethirtyeight')

# Declare constants:
sigma=5.67e-8 #Stefan-Boltzmann constant 
year = np.array([1900, 1950, 2000])
s0 = np.array([1365.0, 1366.5, 1368.])  # Solar forcing in W/m2
t_anom = np.array([-.4, 0, .4])  # Climate change temp anomaly since 1950 in C 

def temp_1layer(s0=1365.0, albedo=0.33):
    '''
    Given solar forcing s0 and albedo, determine the temperature of the Earth's surface 
    using a single-layer perfectly absorbing enegy balanced atomsphere model

    ----------
    s0: float or numpy array, defaults to 1365.0
        Incoming solar irradiance in W/m^2.
    albedo: floating point, defaults to 0.33
        The surface albedo (reflectivity) of the Earth.

    returns
    ------------
    te: float or numpy array
        The temperature of the Earth's surface in Kelvin (K).

    '''

    te = ((1-albedo)*s0 / (2*sigma))**(1/4.)
    return te


def compare_warming():
    '''
    Create a figure to test if changes in solar dricing 
    can account for climate change
    '''

    t_model = temp_1layer(s0=s0)
    t_obs = t_model[1]+ t_anom

    fig, ax=plt.subplots(1,1,figsize=(8,8))

    ax.plot(year, t_obs, label ="Observed Temperature Change") 
    ax.plot(year, t_model, label ="Predicted Temperature Change")

    ax.legend(loc='best')
    ax.set_xlabel('Year')
    ax.set_ylabel('Surface Temperature ($K$)')
    ax.set_title('Can an increase in solar forcing fully explain global warming?')

    fig.tight_layout()
    