from numpy import sqrt

class EffectiveMediumEps:
    def __init__(self, layers):
        """Initialize with a list of tuples (epszz, epsxx, thickness)"""
        self.layers = layers
        self.Ltot = sum(L for epszz, epsxx, L in layers)

    def eps_xx(self):
        """Sum L*epsxx"""
        return sum(epsxx * L for epszz, epsxx, L in self.layers) / self.Ltot

    def eps_zz(self):
        """Inverse(sum L/epszz)"""
        return self.Ltot / sum(L / epszz for epszz, epsxx, L in self.layers)

    def n_xx(self):
        return sqrt(self.eps_xx())

    def n_zz(self):
        return sqrt(self.eps_zz())


class EffectiveMedium:
    def __init__(self, layers, w=None):
        """Initialize with layers information."""
        self.layers = layers
        self.w = w

    def _thickness(self, layer):
        if hasattr(layer, 'd'):
            return layer.d
        elif isinstance(layer, tuple):
            return layer[-1]
        else:
            raise Exception("Cannot determine the thickness of the layer")

    def Ltot(self):
        return sum(self._thickness(layer) for layer in self.layers)

    def eps_xx(self):
        """Sum L*epsxx"""
        Ltot = self.Ltot()
        
        def epsxx(layer):
            if isinstance(layer, tuple):
                if len(layer) == 2:
                    return layer[0]**2
                elif len(layer) == 3:
                    return layer[1]**2
                else:
                    raise Exception("Layer tuple is the wrong length.")
            elif hasattr(layer, 'n'):
                return layer.n(self.w)**2
            else:
                raise Exception("Cannot find a refractive index (xx) for layer")
        
        return sum(epsxx(layer) * self._thickness(layer) for layer in self.layers) / Ltot

    def eps_zz(self):
        """Inverse(sum L/epszz)"""
        Ltot = self.Ltot()
        
        def epszz(layer):
            if isinstance(layer, tuple):
                if len(layer) == 2:
                    return layer[0]**2
                elif len(layer) == 3:
                    return layer[0]**2
                else:
                    raise Exception("Layer tuple is the wrong length.")
            elif hasattr(layer, 'nzz'):
                return layer.nzz(self.w)**2
            elif hasattr(layer, 'n'):
                return layer.n(self.w)**2
            else:
                raise Exception("Cannot find a refractive index (zz) for layer")
        
        return Ltot / sum(self._thickness(layer) / epszz(layer) for layer in self.layers)

    def n_xx(self):
        return sqrt(self.eps_xx())

    def n_zz(self):
        return sqrt(self.eps_zz())


if __name__ == "__main__":
    import transfer_matrix as TM
    from materials import LorentzModel
    from uniaxial_plate2 import AnisoPlate
    import matplotlib.pyplot as plt
    import numpy as np

    pi = np.pi
    freq = np.arange(0, 6e12, 5e9)
    w = 2 * pi * freq
    theta = pi / 4
    eps_b = 1.0

    L = LorentzModel(w=freq, w0=2e12, y=15e10, wp=1.6e12, f=1.0, eps_b=eps_b)

    filterlist = [TM.Layer_eps(eps_b, None)] + \
                 [TM.Layer_eps(L.epsilon(), 5e-6), TM.Layer_eps(eps_b, 40e-6)] * 4 + \
                 [TM.Layer_eps(eps_b, None)]

    f1 = TM.Filter(filterlist, w=w, pol='TM', theta=theta)
    w, R, T = f1.calculate_R_T()

    EM2 = EffectiveMedium(filterlist[1:-1], w)
    anisoslab = AnisoPlate(EM2.n_xx(), EM2.n_zz(), EM2.Ltot(), w, theta, n_b=sqrt(eps_b))

    plt.figure(1)
    THz = freq * 1e-12
    ax1 = plt.subplot(111)
    ax1.plot(THz, R, label="reflection (TM)")
    ax1.plot(THz, T, label="transmission (TM)")
    ax1.plot(THz, anisoslab.R_p(), label="eff. medium: reflection (TM)")
    ax1.plot(THz, anisoslab.T_p(), label="eff. medium: transmission (TM)")
    ax1.legend()
    ax1.set_title("A Uniaxial Transfer Matrix")
    ax1.set_xlabel("Frequency (real) (THz)")
    ax1.set_xlim((1, 6))
    plt.show()
