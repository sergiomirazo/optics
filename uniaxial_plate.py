import numpy as np
from numpy import sin, arcsin, cos, sqrt, exp, pi
import optical_plate as OP
import fresnel as F
from materials import LorentzModel
import matplotlib.pyplot as plt

c = 299792458  # m/s - speed of light

class AnisoInterface(F.Interface):
    """Fresnel equations for uniaxial medium with optical axes perpendicular to the interface."""
    def __init__(self, n1o=1.0, n1e=1.0, n2o=1.0, n2e=1.0, theta=0.0):
        self.n2e = n2e
        self.n1e = n1e
        super().__init__(n1=n1o, n2=n2o, theta=theta)
            
    def _core(self, pol):
        if pol == 'p':
            m = sqrt((1.0 - (self.n2 / self.n2e * sin(self.internal_angle()))**2) / 
                     (1.0 - (self.n1 / self.n1e * sin(self.theta))**2))
            p = self.n2 / self.n1
        elif pol == 's':
            m, p = super()._core('s')
        else:
            raise ValueError("Unknown polarisation")
        return m, p

class AnisoPlate(OP.Plate):
    """Model of a uniaxial material slab with optical axes perpendicular to the slab."""
    def __init__(self, n1o, n1e, d, w, theta, n0=1.0, n2=1.0):
        self.n0 = n0
        self.n1 = n1o
        self.n1e = n1e
        self.n2 = n2
        self.d = d
        self.w = w
        self.theta = theta
        self.i1 = AnisoInterface(n0, n0, n1o, n1e, theta)
        self.i2 = AnisoInterface(n1o, n1e, n2, n2, self.i1.internal_angle())
        
    def _phases(self, pol):
        k = self.n1 * self.w / c
        if pol == 'p':
            costheta = sqrt(1.0 - (self.n1 / self.n1e * sin(self.i1.internal_angle()))**2)
        elif pol == 's':
            costheta = cos(self.i1.internal_angle())
        else:
            raise ValueError("Unknown polarisation")
        phi = 2 * k * costheta * self.d
        expphi = exp(1j * phi)
        expxi = exp(1j * phi / 2)
        return expphi, expxi

def TpSimplified(eps_zz, eps_b, d, w, theta):
    """Simplified formula for P-polarisation transmission of an optically thin uniaxial slab."""
    return 1.0 + (eps_b / eps_zz).imag * sqrt(eps_b) * w / c * sin(theta)**2 / cos(theta) * d

def TpSimplified2(eps_zz, eps_b, d, w, theta):
    """Simplified formula for P-polarisation transmission of an optically thin uniaxial slab."""
    return (1.0 - 0.5 * (eps_b / eps_zz).imag * sqrt(eps_b) * w / c * sin(theta)**2 / cos(theta) * d)**-2

if __name__ == "__main__":
    freq = np.arange(0, 6e12, 5e9)
    L = LorentzModel(w=freq, w0=2e12, y=15e10, wp=1.6e12, f=1.0, eps_b=1.0)

    slab = OP.Plate(n1=L.n(), d=5e-6, w=2*pi*freq, theta=pi/4, n0=1.0, n2=1.0)
    aniso_slab = AnisoPlate(n1o=1.0, n1e=L.n(), d=5e-6, w=2*pi*freq, theta=pi/4, n0=1.0, n2=1.0)

    aniso_slab_Tp = TpSimplified(L.epsilon(), eps_b=1.0, d=5.5e-6, w=2*pi*freq, theta=pi/4)

    plt.figure(1, figsize=(7, 8))
    THz = freq * 1e-12
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)
    
    ax1.plot(THz, L.epsilon().real, label="epsilon real")
    ax1.plot(THz, L.epsilon().imag, label="epsilon imaginary")
    
    ax2.plot(THz, L.n().real, label="refractive index")
    ax2.plot(THz, L.n().imag, label="kappa")
    
    ax3.plot(THz, 2 * (2 * pi * freq) * L.n().imag / c, label="absorption coefficient")
    
    ax4.plot(THz, slab.T_s(), label='Slab Transmission Spol')
    ax4.plot(THz, slab.T_p(), label='Slab Transmission Ppol')
    ax4.plot(THz, aniso_slab.T_s(), label='AnisoSlab Transmission Spol')
    ax4.plot(THz, aniso_slab.T_p(), label='AnisoSlab Transmission Ppol')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(L.w0 * 1e-12, color='r')
        ax.axvline(sqrt(L.w0**2 + L.wp**2 * L.f) * 1e-12, color='r')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.set_title("Various properties of an example Lorentzian Oscillator")
    ax4.set_xlabel("Frequency (real) (THz)")
    ax4.set_xlim((1, 6))
    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.95, top=0.95, hspace=None)
    
    plt.figure(2)
    ax5 = plt.subplot(111)
    ax5.plot(THz, slab.T_s(), label='Isotropic Slab Spol')
    ax5.plot(THz, slab.T_p(), label='Isotropic Slab Ppol')
    ax5.plot(THz, aniso_slab.T_s(), label='Anisotropic Slab Spol')
    ax5.plot(THz, aniso_slab.T_p(), label='Anisotropic Slab Ppol')
    ax5.plot(THz, aniso_slab_Tp, label='Anisotropic Slab Ppol (simplified)')
    
    ax5.set_title("Various properties of an Absorbing Uniaxial Slab")
    ax5.set_xlabel("Frequency (real) (THz)")
    ax5.set_ylabel("Transmission")
    ax5.set_xlim((1, 6))
    ax5.legend(loc=4)
    
    plt.show()
