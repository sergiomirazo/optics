import numpy as np
from transfer_matrix import Filter_base, Filter, interactive, Layer_eps
import matplotlib.pyplot as plt

c = 299792458  # m/s, speed of light
pi = np.pi

class IncoherentFilter(Filter_base):
    def __init__(self, fltrlst, w=None, pol='TE', theta=0.0, *args, **kwargs):
        super().__init__(fltrlst, w=w, pol=pol, theta=theta, *args, **kwargs)

    def _layer_array(self, layer):
        alphad = 2 * self._k_z(layer).imag * layer.d
        nph = np.exp(-alphad)
        pph = np.exp(alphad)
        matrix = np.column_stack((pph, np.zeros_like(self.axis), np.zeros_like(self.axis), nph))
        matrix.shape = (len(self.axis), 2, 2)
        return matrix

    def _interface_array(self, layer_pair):
        mod2 = lambda a: np.abs(a)**2
        lmda = self._lambda(layer_pair)
        matrix = 0.25 * np.column_stack((mod2(1 + lmda), -mod2(1 - lmda), mod2(1 - lmda), mod2(1 + lmda) - 2 * mod2(1 - lmda)))
        matrix.shape = (len(self.axis), 2, 2)
        matrix /= self._lambda(layer_pair).real[:, np.newaxis, np.newaxis]
        return matrix

    def _thin_film(self, filter):
        mod2 = lambda a: np.abs(a)**2
        tm = filter._calculate_M()
        obliquenessfac = filter._lambda((filter[0], filter[-1])).real
        incoherent_matrix = mod2(tm)
        incoherent_matrix[:, 0, 1] *= -1
        incoherent_matrix[:, 1, 1] = (obliquenessfac**2 - mod2(tm[:, 0, 1] * tm[:, 1, 0]).real) / mod2(tm[:, 0, 0])
        incoherent_matrix /= obliquenessfac[:, np.newaxis, np.newaxis]
        return incoherent_matrix

    @interactive
    def _calculate_M(self):
        grammar = [[self[0]]]
        for layer in self[1:-1]:
            if hasattr(layer, 'n') and hasattr(layer, 'd') and hasattr(layer, 'coh') and layer.coh == False:
                grammar[-1].append(layer)
                grammar.append([layer])
                grammar.append([layer])
            elif hasattr(layer, 'n') and hasattr(layer, 'd'):
                grammar[-1].append(layer)
            elif isinstance(layer, list):
                if layer[0].d == None: layer = layer[1:]
                if layer[-1].d == None: layer = layer[:1]
                grammar[-1].extend(layer)
            else:
                raise NameError("unknown type of entry in filter list")
        
        grammar[-1].append(self[-1])
        
        print("Incoherent Transfer Matrix, will process the following grammar")
        for token in grammar: print(token)
        print()
        
        I = np.array(((1.0, 0.0), (0.0, 1.0)))
        tmp = np.array((I,) * len(self.axis))

        for entry in grammar:
            if len(entry) == 2:
                tmp = self._mat_mult(tmp, self._interface_array(entry))
            elif len(entry) == 1:
                tmp = self._mat_mult(tmp, self._layer_array(entry[0]))
            elif len(entry) > 2:
                fltr = Filter(entry, w=self.w, pol=self.pol, theta=self.theta)
                fltrM = self._thin_film(fltr)
                tmp = self._mat_mult(tmp, fltrM)
        
        return tmp

    @interactive
    def calculate_R_T(self):
        axis = self.axis
        M = self._calculate_M()
        T = 1.0 / M[:, 0, 0]
        R = M[:, 1, 0] / M[:, 0, 0]
        return (axis, R.real, T.real)

if __name__ == "__main__":
    f = np.linspace(1.0e10, 10e12, 200)
    
    f1 = IncoherentFilter([
        Layer_eps(1.0, None),
        Layer_eps(3.50, 8.6e-6, coh=False),
        Layer_eps(12.25, None)],
        w=2*pi*f,
        pol='TE',
        theta=0.0)
        
    w, R, T = f1.calculate_R_T()
    w, R2, T2 = f1.calculate_R_T()
    
    ax1 = plt.subplot(111)
    ax1.plot(f, R, label="TM Reflection")
    ax1.plot(f, T, label="TM Transmission")
    ax1.plot(f, R2, label="TE Reflection")
    ax1.plot(f, T2, label="TE Transmission")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_title("Antireflection coating for GaAS or Silicon")
    ax1.legend()
    
    plt.show()
