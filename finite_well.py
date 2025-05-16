import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import simps
from itertools import combinations
from numpy import cos, sin, tan, sqrt, exp, log, pi

print('Libraries imported')

# Define constants
hbar = 1.054571e-34
Me = 9.1093826E-31  # kg
c = 299792458  # m/s
eC = 1.602176e-19  # C
eps0 = 8.8541878176e-12  # F/m
kB = 1.3806504e-23  # J/K

J2meV = 1e3 / eC
meV2J = 1e-3 * eC
dp = 1e11 * 1e4

# Define functions

def rhs(E, Mb, Mw, d, V):
    return sqrt(Mw / Mb * (V / E - 1))
    
def odd_lhs(E, Mb, Mw, d):
    return -1.0 / tan(sqrt(Mw * Me * E / 2.) * d / hbar)

def odd(E, Mb, Mw, d, V):
    return odd_lhs(E, Mb, Mw, d) - rhs(E, Mb, Mw, d, V)
    
def even_lhs(E, Mb, Mw, d):
    return tan(sqrt(Mw * Me * E / 2.) * d / hbar)
    
def even(E, Mb, Mw, d, V):
    return even_lhs(E, Mb, Mw, d) - rhs(E, Mb, Mw, d, V)

# Solve functions - Finding Energy Levels

def EnergyLevels(V, Mb, Mw, d):
    intervals = 300.0
    np.seterr(invalid='ignore')
    
    evenlevels = []
    for E0 in np.linspace(V / intervals, V, intervals, endpoint=False):
        output = fsolve(even, E0, args=(Mb, Mw, d, V), full_output=True)
        level = output[0][0]
        if not evenlevels or abs(level - evenlevels[-1]) / level > 1e-8 and output[2] == 1:
            evenlevels.append(level)

    oddlevels = []
    for E0 in np.linspace(V / intervals, V, intervals, endpoint=False):
        output = fsolve(odd, E0, args=(Mb, Mw, d, V), full_output=True)
        level = output[0][0]
        if not oddlevels or abs(level - oddlevels[-1]) / level > 1e-8 and output[2] == 1:
            oddlevels.append(level)

    levels = np.zeros(len(evenlevels) + len(oddlevels))
    levels[0::2] = evenlevels
    levels[1::2] = oddlevels
    
    return [i * 1e3 / eC for i in [levels, np.array(evenlevels), np.array(oddlevels)]]

def wavefunctions(level, parity, VmeV, Mb, Mw, d):
    f, s = (cos, 1) if parity else (sin, -1)
    k = sqrt(2 * Mw * Me * level * 1e-3 * eC) / hbar
    kappa = sqrt(2 * Mb * Me * (VmeV - level) * 1e-3 * eC) / hbar
    A = 1.0 / sqrt(d / 2.0 + s * sin(k * d) / (2.0 * k) + f(k * d / 2.0)**2 / kappa)
    B = A * f(k * d / 2.0) * exp(kappa * d / 2)
    C = s * B

    def Psi(z):
        a = d / 2.0
        return (z < -a) * C * exp(kappa * z) + (z >= -a) * (z <= a) * A * f(k * z) + (z > a) * B * exp(-kappa * z)
        
    return Psi

# Doping Stuff

def FermiLevel_0K(Ns, levels, Mw):
    Et, Ef = 0.0, 0.0
    Z = hbar**2 * pi / (Mw * Me)
    N2 = Ns * dp
    for i, En in enumerate(levels):
        Et += En
        Efnew = (Z * N2 * J2meV + Et) / (i + 1)
        if Efnew > En:
            Ef = Efnew
        else:
            break
    else:
        print("Cannot be sure that Ef is below next higher energy level.")
    
    Nlevels = (Ef - np.array(levels)) / (Z * J2meV * dp)
    Nlevels *= (Nlevels > 0.0)
    return Ef, Nlevels

def fd2(Ei, Ef, T):
    return kB * T * log(exp((Ef - Ei) / (J2meV * kB * T)) + 1)

def FermiLevel(T, Ns, levels, Mw):
    Z = hbar**2 * pi / (Mw * Me)
    N2 = Ns * dp
    Ef_0K, Nlevels_0K = FermiLevel_0K(Ns, levels, Mw)
    Ef = fsolve(lambda Ef: N2 - sum([fd2(En, Ef, T) for En in levels[:-1]]) / Z, Ef_0K)[0]
    Nlevels = [fd2(En, Ef, T) / (Z * dp) for En in levels]
    return Ef, Nlevels


def transition_generator(seq):
    return combinations(seq, 2)

def _wavefunctions2(level, parity, VmeV, Mb, Mw, d):
    f, s = (cos, 1) if parity else (sin, -1)
    k = sqrt(2 * Mw * Me * level * 1e-3 * eC) / hbar
    kappa = sqrt(2 * Mb * Me * (VmeV - level) * 1e-3 * eC) / hbar
    A = 1.0 / sqrt(d / 2.0 + s * sin(k * d) / (2.0 * k) + f(k * d / 2.0)**2 / kappa)
    return k, kappa, A

def Dipole_matrix(i, f, levels, VmeV, Mb, Mw, d):
    ieven = (i % 2 == 0)
    feven = (f % 2 == 0)
    if ieven == feven:
        return 0.0
    
    dpar = -1 if ieven else 1
    ki, kappai, Ai = _wavefunctions2(levels[i], ieven, VmeV, Mb, Mw, d)
    kf, kappaf, Af = _wavefunctions2(levels[f], feven, VmeV, Mb, Mw, d)
    a = d / 2.0
    ktot = ki + kf
    sin1 = sin(ktot * a)
    cos1 = cos(ktot * a)
    dk = ki - kf
    sin2 = sin(dk * a)
    cos2 = cos(dk * a)
    ikappatot = 1.0 / (kappai + kappaf)
    
    well = (sin1 / ktot**2 - cos1 * a / ktot) + dpar * (sin2 / dk**2 - cos2 * a / dk)
    barrier = (sin1 + dpar * sin2) * (a + ikappatot) * ikappatot
    dipole = Ai * Af * (well + barrier)
    return dipole

def Dipole_matrix_numeric(z, Psi1, Psi2):
    return simps(z * Psi1(z) * Psi2(z), z)

def OscStr(mu_if, w_if, Mw):
    return 2 * Mw * Me * (w_if * 1e-3 * eC) * mu_if**2 / hbar**2

def S_num(z, Psi1, Psi2, w_if, Mw):
    dz = (z[1] - z[0]) * 0.1
    integral = simps((np.gradient(Psi2(z), dz) * Psi1(z) - Psi2(z) * np.gradient(Psi1(z), dz))**2, z)
    return (hbar**2 / (2 * Mw * Me * w_if * meV2J))**2 * integral

def L_eff(w_if, S, Mw):
    return hbar**2 / (2 * S * Mw * Me * w_if * meV2J)


# Main

from transfer_matrix import LayerUniaxial_eps

def QW_GUI(VmeV, Mb, Mw, d, Ns, T, b=2e-7, eps_b=1.0, gammaf=0.1, mag=1.0, figures=False):
    ISBT = LayerUniaxial_eps(eps_b, eps_b, d + b)
    ISBT.VmeV = V
