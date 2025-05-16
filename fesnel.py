import numpy as np
import matplotlib.pyplot as plt

class Interface:
    convention = -1
    
    def __init__(self, n1=1.0, n2=1.0, theta=0.0):
        self.n1 = n1
        self.n2 = n2
        self.theta = np.radians(theta) + 0j
        
    def internal_angle(self):
        return np.arcsin(self.n1 / self.n2 * np.sin(self.theta))
    
    def _core(self):
        m = np.cos(self.internal_angle()) / np.cos(self.theta)
        p = self.n2 / self.n1
        return m, p
    
    def _modsq(self, a):
        return np.abs(a)**2
    
    def r_s(self):
        m, p = self._core()
        return (1.0 - p * m) / (1.0 + p * m)
        
    def r_p(self):
        m, p = self._core()
        return self.convention * (m - p) / (m + p)
        
    def t_s(self):
        m, p = self._core()
        return 2.0 / (1.0 + m * p)
        
    def t_p(self):
        m, p = self._core()
        return 2.0 / (m + p)
        
    def R_s(self):
        return self._modsq(self.r_s()).real
        
    def R_p(self):
        return self._modsq(self.r_p()).real
        
    def T_s(self):
        m, p = self._core()
        return (m * p * self._modsq(self.t_s())).real
        
    def T_p(self):
        m, p = self._core()
        return (m * p * self._modsq(self.t_p())).real
    
    def r_s_b(self):
        return -self.r_s()
        
    def r_p_b(self):
        return -self.r_p()
        
    def t_s_b(self):
        m, p = self._core()
        return 2.0 * m * p / (1.0 + m * p)
        
    def t_p_b(self):
        m, p = self._core()
        return 2.0 * m * p / (m + p)
    
    def R_s_b(self):
        return self.R_s()
        
    def R_p_b(self):
        return self.R_p()
        
    def T_s_b(self):
        m, p = self._core()
        return (1.0 / (m * p) * self._modsq(self.t_s_b())).real
        
    def T_p_b(self):
        m, p = self._core()
        return (1.0 / (m * p) * self._modsq(self.t_p_b())).real

if __name__ == "__main__":
    theta = np.arange(0.0, 90.0, 0.2)
    interface = Interface(n1=1.0, n2=1.5, theta=theta)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(theta, interface.r_s().real, label="r_s")
    plt.plot(theta, interface.r_p().real, label="r_p")
    plt.plot(theta, interface.t_s().real, label="t_s")
    plt.plot(theta, interface.t_p().real, label="t_p")
    plt.legend()
    plt.xlabel("Theta (degrees)")
    plt.title("Reflection and Transmission Coefficients")
    
    plt.subplot(122)
    plt.plot(theta, interface.R_s(), label="R_s")
    plt.plot(theta, interface.R_p(), label="R_p")
    plt.plot(theta, interface.T_s(), label="T_s")
    plt.plot(theta, interface.T_p(), label="T_p")
    plt.legend()
    plt.xlabel("Theta (degrees)")
    plt.title("Reflectance and Transmittance")

    plt.tight_layout()
    plt.show()
