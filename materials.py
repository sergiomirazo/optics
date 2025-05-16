import numpy as np
import matplotlib.pyplot as plt

eps0 = 8.8541E-12  # Farads/metres - vacuum permittivity
m_e = 9.1094e-31   # Kg - mass of electron
q = 1.6022e-19     # C - unit charge
c = 299792458      # m/s - speed of light
pi = np.pi

class Material:
    """Base material class for dielectric, plasma, quantum well, etc."""
    
    def __init__(self):
        if type(self) == Material:
            raise Exception("<Material> must be subclassed.")
        
    def epsilon(self):
        return self.n()**2

    def n(self):
        return np.sqrt(self.epsilon())
        
    def __add__(self, other):
        def new_epsilon():
            return self.epsilon() + other.epsilon()
        newmat = Material()
        newmat.epsilon = new_epsilon.__get__(newmat, Material)
        return newmat

class MaterialEps(Material):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
    
    def epsilon(self):
        return self.eps
    
class Material_nk(Material):
    def __init__(self, nk):
        super().__init__()
        self.nk = nk
        
    def n(self):
        return self.nk

class LorentzModel(Material):
    def __init__(self, w, w0, y, wp, f, eps_b):
        self.w = w
        self.w0 = w0
        self.y = y
        self.wp = wp
        self.f = f
        self.eps_b = eps_b
    
    def epsilon(self):
        w, w0, y, wp, f, eps_b = self.w, self.w0, self.y, self.wp, self.f, self.eps_b
        eps = eps_b * (1 + wp**2 * f / (w0**2 - w**2 - 2j * y * w))
        return eps
        
    @staticmethod   
    def wp(N, meff, eps_b):
        return np.sqrt(N*q**2/(meff*m_e*eps0*eps_b))
    
class DrudeModel(Material):
    def __init__(self, w, y, wp, f, eps_b):
        self.w = w
        self.y = y
        self.wp = wp
        self.f = f
        self.eps_b = eps_b
    
    def epsilon(self):
        w, y, wp, f, eps_b = self.w, self.y, self.wp, self.f, self.eps_b
        eps = eps_b * (1 - wp**2 * f / (w**2 + 2j * y * w))
        return eps
    
    @staticmethod    
    def wp(N, meff, eps_b):
        return np.sqrt(N*q**2/(meff*m_e*eps0*eps_b))

class Metal(Material):
    def __init__(self, w, sigma0, eps_b, simple_n=True):
        self.w = w
        self.sigma0 = sigma0
        self.eps_b = eps_b        
        if not simple_n:
            self.n = super().n
            
    def epsilon(self):
        w, sigma0, eps_b = self.w, self.sigma0, self.eps_b
        eps = eps_b + 1j*sigma0/eps0/w
        return eps
    
    def n(self):
        w, sigma0, eps_b = self.w, self.sigma0, self.eps_b
        p = np.sqrt(sigma0/(2.0*eps0*w))
        return (1 + 1j) * p
    
    @staticmethod    
    def wp(N, meff, eps_b):
        return np.sqrt(N*q**2/(meff*m_e*eps0*eps_b))

    @staticmethod    
    def sigma0(N, meff, y):
        return N*q**2/(meff*m_e*2*y)
        
    @staticmethod    
    def sigma0_b(wp, y, eps_b):
        return wp**2*eps0*eps_b/(2*y)

class Gold(Material):
    def __init__(self, w):
        self.w = w
    
    def epsilon(self):
        w = self.w
        epsinf = 1.53
        wp = 12.9907004642E15
        Gammap = 110.803033371e12
        C1 = 3.78340272066E15
        w1 = 4.02489651134E15
        Gamma1 = 0.818978942308E15
        C2 = 7.73947471764E15
        w2 = 5.69079023356E15
        Gamma2 = 2.00388464607E15
        sr2 = np.sqrt(2)
        G1 = C1 * ((1 - 1j) / sr2 / (w1 - w - 1j * Gamma1) + (1 + 1j) / sr2 / (w1 + w + 1j * Gamma1))
        G2 = C2 * ((1 - 1j) / sr2 / (w2 - w - 1j * Gamma2) + (1 + 1j) / sr2 / (w2 + w + 1j * Gamma2))
        eps = epsinf - wp**2/(w**2 + 1j*w*Gammap) + G1 + G2
        return eps        

class GaAs_THz(Material): 
    def __init__(self, w, n=0.0):
        self.w = w
        self.doping = n
        
    def epsilon(self):
        w = self.w * 1e-12
        eps = 10.4 + 5161.4 / (2620.0 - w**2 - 0.2j * w)
        n = self.doping
        if n != 0.0:
            T = 10.0 / (1.0 - 2.0 * np.log10(n)) if n < 1.0 else 10.0
            eps -= 47436.84 * n / (w * (w + 1j * T))
        return eps   

if __name__ == "__main__":
    plt.figure(1)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    w = np.arange(0, 5e12, 5e9)
    L = LorentzModel(w=w, w0=1e12, y=5e10, wp=8e11, f=0.96, eps_b=1.0)
    ax1.plot(w, L.epsilon().real, label="epsilon real")
    ax1.plot(w, L.epsilon().imag, label="epsilon imaginary")
    ax2.plot(w, L.n().real, label="refractive index")
    ax2.plot(w, L.n().imag, label="kappa")
    ax3.plot(w, 2 * w * L.n().imag / c, label="absorption coefficient")
    for ax in (ax1, ax2, ax3):
        ax.axvline(L.w0)
        ax.axvline(np.sqrt(L.w0**2 - L.y**2))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Various properties of an example Lorentzian Oscillator")
    ax3.set_xlabel("Frequency (real) (Hz)")
    ax3.text(1.4e12, 4000, "It is interesting that the absorption coefficient plotted here is \n \
nothing like the profile, we would see if we modelled the \n \
absorption of a slab of this material")
    
    plt.figure(2)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    w = np.arange(0, 5e12, 5e9)
    D = DrudeModel(w=w, y=5e10, wp=8e11, f=0.96, eps_b=1.0)
    ax1.plot(w, D.epsilon().real, label="epsilon real")
    ax1.plot(w, D.epsilon().imag, label="epsilon imaginary")
    ax2.plot(w, D.n().real, label="refractive index")
    ax2.plot(w, D.n().imag, label="kappa")
    ax3.plot(w, 2 * w * D.n().imag / c, label="absorption")
    for ax in (ax1, ax2, ax3):
        ax.axvline(D.wp)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Various properties of an example Drude model")
    ax3.set_xlabel("Frequency (real) (Hz)")
    ax3.text(1.4e12, 2000, "It is interesting that the absorption coefficient plotted here is \n \
nothing like the profile, we would see if we modelled the \n \
absorption of a slab of this material")    
    
    plt.figure(3)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex
