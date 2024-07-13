"""
Created on Sat July 13 2024
@author: ltakiguchi

analytical closed form expressions for the PSD of a bead trajectory
adapted from tweezepy docs

functions:  
lansdorpPSD(f,fs,g,k,e,kT=4.1): PSD eq 7 from lansdorp et al 2012 (aliasing + lowpass filtering)
aliasPSD(f,fs,a,k,e,kT=4.1): PSD eq 8 from lansdorp et al 2012 (aliasing)

"""

import autograd.numpy as np

def lansdorpPSD(f,fs,g,k,e,kT = 4.1):
    """
    Analytical function for the PSD of a trapped bead with aliasing and lowpass filtering.
    Eq. 7 in Lansdorp et al. (2012).

    Parameters
    ----------
    f : array-like
        frequency.
    fs : float
        Acquisition frequency.
    g : float
        drag coefficient.
    k : float
        spring constant.
    kT : float
        Thermal energy in pN nm, defaults to 4.1

    Returns
    -------
    PSD : array
        theoretical power spectral density.
    """
    tc = g/k
    fc = 1./tc
    PSD = 2.*kT*tc/k * (1. + 2.*tc*fs*np.sin(np.pi*f/fs)**2 * np.sinh(fc/fs)/(np.cos(2.*np.pi*f/fs) - np.cosh(fc/fs))) + pow(e,2)/fs
    return PSD
    

def aliasPSD(f,fs,a,k,e, kT = 4.1):
    """
    Analytical function for the PSD of a trapped bead with aliasing.
    Eq. 8 in Lansdorp et al. (2012).

    Parameters
    ----------
    f : array-like
        frequency.
    fs : float
        Acquisition frequency.
    a : float
        alpha.
    k : float
        kappa.
    kT : float
        Thermal energy in pN nm. Default value is 4.1

    Returns
    -------
    PSD : array
        theoretical power spectral density.
    """
    kT = 4.1 # thermal energy in pN*nm
    return kT/(k*fs) * (np.sinh(k/(a*fs))/(np.cosh(k/(a*fs))-np.cos(2*np.pi*f/fs))) + pow(e,2)/fs