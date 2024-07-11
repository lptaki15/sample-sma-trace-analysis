"""
Created on Thurs June 9 2024
@author: ltakiguchi

functions: 
sigma_localizations()
sigma_brownian_theoretical()
sigma_brownian_loc_theoretical()
"""

import numpy as np

# frequency-indepedent contribution of photon-limited localization uncertainty -> sigma of localizations
def sigma_localizations(mu_intensity): # input: mean intensity of trajectory
    sigma_loc_error = (981.1868/np.sqrt(mu_intensity)) - 6.2673 # in nm, equation coefficients determined by bead fit
    return sigma_loc_error

# theoretical sigma of θ (brownian motion only)
def sigma_brownian_theoretical(t,kbT,kappa_axial,tau_axial): # input: time vector
    sigma_theoretical_brownian = np.sqrt(
        (2*kbT/kappa_axial)* # constant factor
        (tau_axial/t - (tau_axial**2/t**2)* # time dependent factor
        (1-np.exp(-t/tau_axial)))) # exponential decay factor
    return sigma_theoretical_brownian

# theoretical sigma of θ (brownian motion + localization uncertainty)
def sigma_brownian_loc_theoretical(t, sigma_loc,kbT,kappa_axial,tau_axial,fsample): # input: time vector, sigma of localizations
    sigma_theoretical_brownian_loc = np.sqrt(
        (2*kbT/kappa_axial)* # constant factor
        (tau_axial/t - (tau_axial**2/t**2)* # time dependent factor
        (1-np.exp(-t/tau_axial)))+ # exponential decay factor
        sigma_loc**2/(fsample*t)) # localization uncertainty factor
    return sigma_theoretical_brownian_loc