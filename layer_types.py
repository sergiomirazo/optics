import numpy as np
import os
from scipy.interpolate import splrep, splev

eps0 = 8.8541E-12  # Farads/metres - vacuum permittivity
m_e = 9.1094e-31   # Kg - mass of electron
q = 1.6022e-19     # C - unit charge
c = 299792458      # m/s - speed of light
pi = np.pi

class MaterialLayer:
    """Base class for materials like dielectric, plasma, quantum well, etc."""
    
    def __init__(self, d, coh):
        if type(self) == MaterialLayer:
            raise Exception("<MaterialLayer> must be subclassed.")
        self.d = d
        self.coh = coh
        
    def epsilon(self, w):
        return self.n(w)**2
    
    def n(self, w):
        return np.sqrt(self.epsilon(w))

    def __add__(self, other):
        def new_epsilon(self2, w):
            return self.epsilon(w) + other.epsilon(w)
        newmat = MaterialLayer()
        newmat.epsilon = new_epsilon.__get__(newmat, MaterialLayer)
        return newmat
    
    def __repr__(self):
        return f"Layer({self.n(self.d)}, {self.d}, coh={self.coh})"
        
class AnisoMaterialLayer(MaterialLayer):
    
    def epsilonzz(self, w):
        return self.nzz(w)**2
    
    def nzz(self, w):
        return np.sqrt(self.epsilonzz(w))

class MaterialEps(MaterialLayer):
    """Initialize using material_eps(epsilon). epsilon can be a (complex) number or array."""
    def __init__(self, eps, d, coh=True):
        super().__init__(d, coh)
        self.eps = eps
    
    def epsilon(self, w):
        return self.eps
    
class Material_nk(MaterialLayer):
    """Initialize using material_nk(nk). nk can be a (complex) number or array."""
    def __init__(self, nk, d, coh=True):
        super().__init__(d, coh)
        self.nk = nk
        
    def n(self, w):
        return self.nk

class SopraLayer(MaterialLayer):
    """Loads SOPRA data files and provides an interpolated refractive index function."""
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SOPRA data")
    
    def __init__(self, matname, d, coh=True):
        super().__init__(d, coh)
        self.name = matname
        with open(os.path.join(self.directory, f'{matname}.MAT')) as fobj:
            output = []
            for line in fobj:
                linelist = line.split('*')
                if linelist[0] == 'DATA1':
                    output.append(list(map(float, linelist[2:5])))
            data = np.array(output)
        
        w = c * 2 * pi * 1e9 / data[:, 0]
        nkspline_real = splrep(w[::-1], data[::-1, 1], s=0)
        nkspline_imag = splrep(w[::-1], data[::-1, 2], s=0)
        
        self._n = lambda axis: splev(axis, nkspline_real, der=0) + 1j * splev(axis, nkspline_imag, der=0)
        self.wupper = max(w)
        self.wlower = min(w)
    
    def n(self, w):
        if max(w) > self.wupper or min(w) < self.wlower:
            raise Exception(f"{self.name}: Frequency range outside of material's data range")
        return self._n(w)

    def __repr__(self):
        return f"Layer({self.name}, {self.d}, coh={self.coh})"

class LorentzModel(MaterialLayer):
    def __init__(self, w0, y, wp, f, eps_b, d, coh=True):
        super().__init__(d, coh)
        self.w0 = w0
        self.y = y
        self.wp = wp
        self.f = f
        self.eps_b = eps_b
    
    def epsilon(self, w):
        w0, y, wp, f, eps_b = self.w0, self.y, self.wp, self.f, self.eps_b
        eps = eps_b * (1 + wp**2 * f / (w0**2 - w**2 - 2j * y * w))
        return eps
        
    @staticmethod   
    def wp(N, meff, eps_b):
        return np.sqrt(N * q**2 / (meff * m_e * eps0 * eps_b))
    
class DrudeModel(MaterialLayer):
    def __init__(self, y, wp, f, eps_b, d, coh=True):
        super().__init__(d, coh)
        self.y = y
        self.wp = wp
        self.f = f
        self.eps_b = eps_b
    
    def epsilon(self, w):
        y, wp, f, eps_b = self.y, self.wp, self.f, self.eps_b
        eps = eps_b * (1 - wp**2 * f / (w**2 + 2j * y * w))
        return eps
    
    @staticmethod    
    def wp(N, meff, eps_b):
        return np.sqrt(N * q**2 / (meff * m_e * eps0 * eps_b))

class Metal(MaterialLayer):
    def __init__(self, sigma0, eps_b, d, coh=True, simple_n=True):
        super().__init__(d, coh)
        self.sigma0 = sigma0
        self.eps_b = eps_b        
        if not simple_n:
            self.n = super(Metal, self).n
            
    def epsilon(self, w):
        sigma0, eps_b = self.sigma0, self.eps_b
        eps = eps_b + 1j * sigma0 / eps0 / w
        return eps
    
    def n(self, w):
        sigma0, eps_b = self.sigma0, self.eps_b
        p = np.sqrt(sigma0 / (2.0 * eps0 * w))
        return (1 + 1j) * p
    
    @staticmethod    
    def wp(N, meff, eps_b):
        return np.sqrt(N * q**2 / (meff * m_e * eps0 * eps_b))

    @staticmethod    
    def sigma0(N, meff, y):
        return N * q**2 / (meff * m_e * 2 * y)
        
    @staticmethod    
    def sigma0_b(wp, y, eps_b):
        return wp**2 * eps0 * eps_b / (2 * y)

class Gold(MaterialLayer):
    def __init__(self, d, coh=True):
        super().__init__(d, coh)
        
    def epsilon(self, w):
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
        G1 = C1 * ((1-1j)/sr2/(w1 - w - 1j*Gamma1) + (1+1j)/sr2/(w1 + w + 1j*Gamma1))
        G2 = C2 * ((1-1j)/sr2/(w2 - w - 1j*Gamma2) + (1+1j)/sr2/(w2 + w + 1j*Gamma2))
        eps = epsinf - wp**2/(w**2 + 1j*w*Gammap) + G1 + G2
        
        return eps        

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    w = np.arange(0, 5e12, 5e9)
    L = LorentzModel(w0=1e12, y=5e10, wp=8e11, f=0.96, eps_b=1.0, d=None)
    ax1.plot(w, L.epsilon(w).real, label="epsilon real")
    ax1.plot(w, L.epsilon(w).imag, label="epsilon imaginary")
    ax2.plot(w, L.n(w).real, label="refractive index")
    ax2.plot(w, L.n(w).imag, label="kappa")
    ax3.plot(w, 2 * w * L.n(w).imag / c, label="absorption coefficient")
    for ax in (ax1, ax2, ax3):
        ax.axvline(L.w0)
        ax.axvline(np.sqrt(L.w0**2 - L.y**2))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title("Properties of a Lorentzian
