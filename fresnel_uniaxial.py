import numpy as np
import matplotlib.pyplot as plt

class Interface:
    convention = -1
    
    def __init__(self, eps1xx=1.0, eps1zz=1.0, eps2xx=1.0, eps2zz=1.0, kx=0.0):
        self.eps1xx = eps1xx
        self.eps1zz = eps1zz
        self.eps2xx = eps2xx
        self.eps2zz = eps2zz
        self.kx = kx + 0j

    def _core(self, pol):
        eps1xx, eps1zz = self.eps1xx, self.eps1zz
        eps2xx, eps2zz = self.eps2xx, self.eps2zz
        
        k1z = np.sqrt(eps1xx * eps1zz - self.kx**2)
        k2z = np.sqrt(eps2xx * eps2zz - self.kx**2)

        if pol == 'p':
            Lambda = k2z * eps1xx / (k1z * eps2xx)
        elif pol == 's':
            Lambda = k2z / k1z
        return Lambda
        
    def _modsq(self, a):
        return np.abs(a)**2

    def r_s(self):
        Lambda = self._core('s')
        return (1.0 - Lambda) / (1.0 + Lambda)
        
    def r_p(self):
        Lambda = self._core('p')
        return self.convention * (1.0 - Lambda) / (1.0 + Lambda)
        
    def t_s(self):
        Lambda = self._core('s')
        return 2.0 / (1.0 + Lambda)
        
    def t_p(self):
        Lambda = self._core('p')
        return 2.0 / (1.0 + Lambda) * np.sqrt(self.eps1xx / self.eps2xx)
        
    def R_s(self):
        return self._modsq(self.r_s())
        
    def R_p(self):
        return self._modsq(self.r_p())
        
    def T_s(self):
        Lambda = self._core('s')
        return Lambda * self._modsq(self.t_s())
        
    def T_p(self):
        Lambda = self._core('p')
        return 4 * Lambda / (1 + Lambda)**2


if __name__ == "__main__":
    theta = np.arange(0.0, 90.0, 0.2)
    theta2 = theta * np.pi / 180.0
    I = Interface(eps1xx=1.0, eps1zz=1.0, eps2xx=2.25, eps2zz=1.0, kx=0.5)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(theta, I.r_s().real, label="r_s")
    plt.plot(theta, I.r_p().real, label="r_p")
    plt.plot(theta, I.t_s().real, label="t_s")
    plt.plot(theta, I.t_p().real, label="t_p")
    plt.legend()
    plt.xlabel("Theta (degrees)")
    plt.title("Reflection and Transmission Coefficients")
    
    plt.subplot(122)
    plt.plot(theta, I.R_s(), label="R_s")
    plt.plot(theta, I.R_p(), label="R_p")
    plt.plot(theta, I.T_s(), label="T_s")
    plt.plot(theta, I.T_p(), label="T_p")
    plt.legend()
    plt.xlabel("Theta (degrees)")
    plt.title("Reflectance and Transmittance")

    plt.tight_layout()
    plt.show()
