import numpy as np
from numpy import sin, arcsin, cos, exp
from fresnel import Interface
import matplotlib.pyplot as plt

c = 299792458  # m/s - speed of light

class Plate:
    def __init__(self, n1, d, w, theta, n0=1.0, n2=1.0):
        self.n0 = n0  # Refractive index of initial medium
        self.n1 = n1  # Refractive index of plate
        self.n2 = n2  # Refractive index of exit medium
        self.d = d  # Thickness of plate (m)
        self.w = w  # Natural frequency (Hz)
        self.theta = theta + 0j  # Angle of incidence (rad)
        self.i1 = Interface(n0, n1, theta)  # First interface
        self.i2 = Interface(n1, n2, self.i1.internal_angle())  # Second interface
        
    def _phases(self, pol):
        k = self.n1 * self.w / c
        phi = 2 * k * cos(self.i1.internal_angle()) * self.d
        expphi = exp(1j * phi)
        expxi = exp(1j * phi / 2)
        return expphi, expxi
    
    def r_s(self):
        r01 = self.i1.r_s()
        t01 = self.i1.t_s()
        r12 = self.i2.r_s()
        r10 = -r01
        t10 = self.i1.t_s_b()
        expphi, expxi = self._phases('s')
        return r01 + (r12 * t01 * t10 * expphi) / (1 - r10 * r12 * expphi)
    
    def r_p(self):
        r01 = self.i1.r_p()
        t01 = self.i1.t_p()
        r12 = self.i2.r_p()
        r10 = -r01
        t10 = self.i1.t_p_b()
        expphi, expxi = self._phases('p')
        return r01 + (r12 * t01 * t10 * expphi) / (1 - r10 * r12 * expphi)
    
    def t_s(self):
        t01 = self.i1.t_s()
        r12 = self.i2.r_s()
        t12 = self.i2.t_s()
        r10 = -self.i1.r_s()
        expphi, expxi = self._phases('s')
        return (t01 * t12 * expxi) / (1 - r10 * r12 * expphi)
        
    def t_p(self):
        t01 = self.i1.t_p()
        r12 = self.i2.r_p()
        t12 = self.i2.t_p()
        r10 = -self.i1.r_p()
        expphi, expxi = self._phases('p')
        return (t01 * t12 * expxi) / (1 - r10 * r12 * expphi)

    def _modsq(self, a):
        return abs(a)**2

    def R_s(self):
        return self._modsq(self.r_s())
    
    def R_p(self):
        return self._modsq(self.r_p())
    
    def T_s(self):
        t_s = self.t_s()
        m = self.n2 / self.n0
        p = cos(self.i2.internal_angle()) / cos(self.theta)
        return (m * p * self._modsq(t_s)).real
    
    def T_p(self):
        t_p = self.t_p()
        m = self.n2 / self.n0
        p = cos(self.i2.internal_angle()) / cos(self.theta)
        return (m * p * self._modsq(t_p)).real

if __name__ == "__main__":
    pi = np.pi
    f = np.arange(100e12, 1500e12, 5e10)
    w = 2 * pi * f
    
    plt.figure()
    ax = plt.subplot(111)
    
    slab = Plate(n1=3.0 + 0.05j, d=1e-6, w=w, theta=0.0, n0=1.0, n2=3.0 + 0j)
    ax.plot(f, slab.T_s(), label='T_s')
    ax.plot(f, slab.T_p(), label='T_p')   
    
    slab = Plate(n1=3.0 + 0.05j, d=1e-6, w=w, theta=0.0, n0=1.0, n2=3.0 + 10j)
    ax.plot(f, slab.T_s(), label='T_s n2=10+10j')
    ax.plot(f, slab.T_p(), label='T_p n2=10+10j') 
    
    ax.legend()
    plt.show()
