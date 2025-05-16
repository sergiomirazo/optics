import numpy as np
from numpy import sin, arcsin, cos, exp, sqrt, pi
from materials import QW_ISBT, Material_nk
import optical_plate as OP
import matplotlib.pyplot as plt

c = 299792458  # m/s - speed of light

class AnisoPlateEps:
    def __init__(self, eps_xx, eps_zz, d, w, theta, eps_b=1.0):
        self.epsb = eps_b
        self.epszz = eps_zz
        self.epsxx = eps_xx
        self.d = d
        self.w = w
        self.theta = theta
        
    def _shared(self, pol):
        epsb, nxx, d, w, theta = self.epsb, sqrt(self.epsxx), self.d, self.w, self.theta
        K = w / c
        k0z = sqrt(epsb) * K * cos(theta)

        if pol == 'p':
            k1z = nxx * K * sqrt(1 - epsb * sin(theta)**2 / self.epszz)
        elif pol == 's':
            k1z = K * sqrt(self.epsxx - epsb * sin(theta)**2)
        delta = k1z * d
        return delta, k0z, k1z
    
    def r_s(self):
        delta, k0z, k1z = self._shared(pol='s')
        Lambda = k1z / k0z
        r01 = (1 - Lambda) / (1 + Lambda)
        exp2delta = exp(2j * delta)
        return r01 * (1 - exp2delta) / (1 - r01**2 * exp2delta)
    
    def r_p(self):
        delta, k0z, k1z = self._shared(pol='p')
        Lambda = self.epsb * k1z / (self.epsxx * k0z)
        r01 = (1 - Lambda) / (1 + Lambda)
        exp2delta = exp(2j * delta)
        return r01 * (1 - exp2delta) / (1 - r01**2 * exp2delta)
    
    def t_s(self):
        delta, k0z, k1z = self._shared(pol='s')
        return 1.0 / (cos(delta) - 1j * sin(delta) * (k0z**2 + k1z**2) / (2 * k0z * k1z))
        
    def t_p(self):
        delta, k0z, k1z = self._shared(pol='p')
        t_p = 1.0 / (cos(delta) - 1j * sin(delta) * ((self.epsxx * k0z)**2 + (self.epsb * k1z)**2) /
                     (2 * self.epsb * self.epsxx * k0z * k1z))
        return t_p
    
    def _modsq(self, a):
        return abs(a)**2
            
    def R_s(self):
        return self._modsq(self.r_s())
    
    def R_p(self):
        return self._modsq(self.r_p())
    
    def T_s(self):
        return self._modsq(self.t_s())
    
    def T_p(self):
        return self._modsq(self.t_p())

class AnisoPlate(AnisoPlateEps):
    def __init__(self, n_xx, n_zz, d, w, theta, n_b=1.0):
        epsb = n_b**2
        epszz = n_zz**2
        epsxx = n_xx**2
        super().__init__(epsxx, epszz, d, w, theta, eps_b=epsb)

if __name__ == "__main__":
    theta = pi / 4
    f = np.arange(1e10, 4e12, 1e10)
    w = 2 * pi * f

    well = Material_nk(3.585)
    barrier = Material_nk(3.585)
    
    w0 = 1.1e12 * 2 * pi
    y = 1e11 * 2 * pi
    f12 = 0.96
    Lqw = 47e-9
    N2D = 1.28e11
    N = N2D / Lqw / 100.0
    eps_well = well.epsilon()
    wp = QW_ISBT.wp(N, eps_well, meff=0.067)
    
    ISBT1 = QW_ISBT(w, w0, y, f12, wp, well.epsilon())
    QW0 = OP.Plate(ISBT1.n(), Lqw, w, theta, n0=barrier.n(), n2=barrier.n())
    
    def nsimplified(eps):
        return well.n() + 1j * eps.imag / (2 * well.n().real)
    QWsimplified = OP.Plate(nsimplified(ISBT1.epsilon()), Lqw, w, theta, n0=barrier.n(), n2=barrier.n())

    n_zz = ISBT1.n()
    n_xx = well.n()
    QW = AnisoPlate(n_xx, n_zz, Lqw, w, theta, n_b=barrier.n())
    
    ax = plt.subplot(111)
    
    ax.axvline(w0 / (2 * pi), label='w0')
    ax.axvline(sqrt(w0**2 - y**2) / (2 * pi), label='sqrt(w0^2 - y^2)')
    ax.plot(f, ISBT1.epsilon().imag * 0.0003, label="imaginary epsilon")
    ax.plot(f, w * ISBT1.epsilon().imag / well.n().real / c * Lqw / cos(theta), label="simplified absorption")
    ax.plot(f, 2 * w * ISBT1.n().imag / c * Lqw / cos(theta), label="proper absorption")
    ax.plot(f, 1 - QWsimplified.T_p(), label="simplified dielectric absorption")
    ax.plot(f, 1 - QW0.T_p(), label="absorption in a slab")
    ax.plot(f, 1 - QW.T_p(), label="anisotropic slab absorption")
    
    ax.axvline(sqrt(w0**2 + f12 * wp**2) / (2 * pi), label='sqrt(w0^2 + f12 * wp^2)')
    ax.axvline(sqrt(w0**2 - y**2 + f12 * wp**2) / (2 * pi), label='sqrt(w0^2 - y^2 + f12 * wp^2)')
    
    ax.legend()
    plt.show()
