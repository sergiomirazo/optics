import numpy as np
from numpy.random import random_sample
from numbers import Number
import copy
import matplotlib.pyplot as plt

c = 299792458  # m/s
pi = np.pi

def param_check(param, w):
    if hasattr(param, 'n'):
        param = param.n()
    if callable(param):
        x = param(w)
    elif hasattr(param, '__len__'):
        if len(param) != len(w):
            raise NameError("w and parameter array are not compatible")
        else:
            x = param
    elif isinstance(param, Number):
        x = np.repeat(param, len(w))
    else:
        raise TypeError("Don't know how to handle this input")
    return x

class Layer:
    def __init__(self, n, d, coh=True):
        self._n = n
        self.d = d
        self.coh = coh

    def n(self, w):                
        return param_check(self._n, w)      
    
    def __repr__(self):
        return f"Layer({repr(self._n)}, {repr(self.d)}, coh={repr(self.coh)})"

class LayerUniaxial:
    def __init__(self, nxx, nzz, d, coh=True):
        self._nxx = nxx
        self._nzz = nzz
        self.d = d
        self.coh = coh
        
    def n(self, w):
        return param_check(self._nxx, w)
    
    def nzz(self, w):
        return param_check(self._nzz, w)
        
    def __repr__(self):
        return f"LayerUniaxial({repr(self._nxx)}, {repr(self._nzz)}, {repr(self.d)}, coh={repr(self.coh)})"
    
class Layer_eps:
    def __init__(self, eps, d, coh=True):
        self.eps = eps
        self.d = d
        self.coh = coh
        
    def n(self, w):
        return np.sqrt(param_check(self.eps, w))
        
    def __repr__(self):
        return f"Layer_eps({repr(self.eps)}, {repr(self.d)}, coh={repr(self.coh)})"

class LayerUniaxial_eps:
    def __init__(self, epsxx, epszz, d, coh=True):
        self.epsxx = epsxx
        self.epszz = epszz
        self.d = d
        self.coh = coh
    
    def n(self, w):
        return np.sqrt(param_check(self.epsxx, w))

    def nzz(self, w):
        return np.sqrt(param_check(self.epszz, w))
        
    def __repr__(self):
        return f"LayerUniaxial_eps({repr(self.epsxx)}, {repr(self.epszz)}, {repr(self.d)}, coh={repr(self.coh)})"

def interactive(func):
    def wrapping(self, *args, **kwargs):
        kwargsA = {k: kwargs[k] for k in kwargs if k not in self.__dict__}
        kwargsB = {k: kwargs[k] for k in kwargs if k in self.__dict__}
        SaveState = copy.copy(self.__dict__)
        try:
            for k in kwargsB:
                print(f'Temporarily changing {k} for this calculation')
                setattr(self, k, kwargsB[k])
            self._checks_n_axis()
            self.n0sinth2 = (self[0].n(self.w) * np.sin(self.theta))**2
            return func(self, *args, **kwargsA)
        finally:
            self.__dict__ = SaveState
    return wrapping

class Filter_base(list):
    def __init__(self, fltrlst, w=None, pol='TE', theta=0.0, *args, **kwargs):
        super().__init__(fltrlst, *args, **kwargs)
        self.w = w
        self.pol = pol
        self.theta = theta
        self._checks_n_axis()
        self.n0sinth2 = (fltrlst[0].n(self.w) * np.sin(self.theta))**2

    def _checks_n_axis(self):
        w = self.w
        theta = self.theta
        if w is None:
            w = np.array([1.0])
        elif isinstance(w, Number):
            w = np.array([w])
        elif hasattr(w, '__iter__'):
            w = np.array(w)
        self.w = w

        if isinstance(theta, Number):
            theta = np.array([theta])
        elif hasattr(theta, '__iter__'):
            theta = np.array(theta)
        self.theta = theta

        if hasattr(theta, '__iter__') and len(theta) > 1:
            if len(theta) == len(w):
                self.axis = w
                print('Exceptionally both angle and frequency are arrays with the same length')
                print('but setting self.axis to frequency')
            else:
                assert len(w) == 1, "Cannot vary both frequency and angle at the same time"
                self.axis = theta
        else:
            self.axis = w
        
    def __repr__(self):
        return (repr(self[:]) + 
                f", w= {repr(self.w)}, pol= {repr(self.pol)}, theta (radians) = {repr(self.theta)}, theta (degrees) = {repr(self.theta * 180 / pi)}")
    
    def __str__(self):
        globals = f"w : {repr(self.w)}\npol : {repr(self.pol)}\ntheta : {repr(self.theta)} theta(deg): {repr(self.theta * 180 / pi)}"
        stack = "[\n" + ",\n".join([repr(l) for l in self]) + "\n]"
        return globals + '\n' + stack
        
    def _lambda(self, layer_pair):
        if hasattr(layer_pair[1], 'nzz') and self.pol == 'TM':
            n1 = layer_pair[1].nzz
        else:
            n1 = layer_pair[1].n
        
        if hasattr(layer_pair[0], 'nzz') and self.pol == 'TM':
            n0 = layer_pair[0].nzz
        else:
            n0 = layer_pair[0].n
        
        w = self.w
        cos_ratio = np.sqrt(1 + 0j - self.n0sinth2 / n1(w)**2)
        cos_ratio = cos_ratio / np.sqrt(1 + 0j - self.n0sinth2 / n0(w)**2)
        n_ratio = layer_pair[1].n(w) / layer_pair[0].n(w)
        
        if self.pol == 'TE':
            lmda = cos_ratio * n_ratio
        elif self.pol == 'TM':
            lmda = cos_ratio / n_ratio
        else: 
            raise NameError("_lambda(): pol should be 'TE' or 'TM'")
        return lmda
    
    def _k_z(self, layer):
        if hasattr(layer, 'nzz') and self.pol == 'TM':
            n1 = layer.nzz
        else:
            n1 = layer.n
            
        w = self.w
        k = layer.n(w) * w / c
        costh = np.sqrt(1 + 0j - self.n0sinth2 / n1(w)**2)
        
        return k * costh

    def _phase(self, layer):
        return self._k_z(layer) * layer.d
    
    def _mat_mult(self, A, B):
        return np.sum(np.transpose(A, (0, 2, 1))[:, :, :, np.newaxis] * B[:, :, np.newaxis, :], axis=-3)

class Filter(Filter_base):
    def __init__(self, fltrlst, w=None, pol='TE', theta=0.0, *args, **kwargs):
        super().__init__(fltrlst, w=w, pol=pol, theta=theta, *args, **kwargs)
        
    def _layer_array(self, layer):
        phase = self._k_z(layer) * layer.d
        if hasattr(layer, 'coh') and isinstance(layer.coh, float):
            coh = layer.coh
            nph = np.exp(-1j * (phase + coh * (2 * random_sample(self.axis.shape) - 1)))
            pph = np.exp(1j * (phase + coh * (2 * random_sample(self.axis.shape) - 1)))
        else:
            nph = np.exp(-1j * phase)
            pph = np.exp(1j * phase)
        matrix = np.column_stack((nph, np.zeros_like(self.axis), np.zeros_like(self.axis), pph))
        matrix.shape = (len(self.axis), 2, 2)
        return matrix
        
    def _interface_array(self, layer_pair):
        lmda = self._lambda(layer_pair)
        matrix = 0.5 * np.column_stack(((1 + lmda), (1 - lmda), (1 - lmda), (1 + lmda)))
        matrix.shape = (len(self.axis), 2, 2)
        return matrix
        
    def _layer_array2(self, layer_pair):
        phase = self._k_z(layer_pair[1]) * layer_pair[1].d
        if hasattr(layer_pair[1], 'coh') and isinstance(layer_pair[1].coh, float):
            coh = layer_pair[1].coh
            nph = np.exp(-1j * (phase + coh * (2
