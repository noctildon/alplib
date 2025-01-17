"""
Define form factors
"""

from alplib.materials import Material
from .constants import *
from .fmath import *
from scipy.special import spherical_jn




def nuclear_ff(t, m, z, a):
    # Parameterization of the coherent nuclear form factor (Tsai 1986, B49)
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    # a: number of nucleons
    return (2*m*z**2) / (1 + t / 164000*np.power(a, -2/3))**2




def atomic_elastic_ff(t, z):
    # Coherent atomic form factor parameterization (Tsai 1986, B38)
    # Fit based on Thomas-Fermi model
    # t: MeV
    # z: atomic number
    a = 184.15*np.power(2.718, -1/2)*np.power(z, -1/3) / M_E
    return (z*t*a**2)**2 / (1 + t*a**2)**2



class AtomicElasticFF:
    """
    square of the form factor
    """
    def __init__(self, z):
        self.z = z

    def __call__(self, q):
        t = q**2
        a = 184.15*np.power(2.718, -1/2)*np.power(self.z, -1/3) / M_E
        return power(self.z*(t*a**2) / (1 + t*a**2), 2)





class ElectronElasticFF:
    """
    square of the form factor
    """
    def __init__(self, material: Material):
        self.z = material.z
        self.frac = material.frac

    def __call__(self, q):
        t = q**2
        a = 184.15*np.power(2.718, -1/2)*np.power(self.z, -1/3) / M_E
        return np.dot(self.frac, power(self.z*(t*a**2) / (1 + t*a**2) - self.z, 2))




class NuclearHelmFF:
    """
    square of the Helm nuclear form factor
    """
    def __init__(self, n, z):
        self.s = 0.9 * (10 ** -15) / METER_BY_MEV
        self.r1 = sqrt((1.23*power(n+z, 1/3) - 0.6)**2 - 5*0.9**2 + 7*power(pi*0.52, 2)/3) * (10 ** -15) / METER_BY_MEV
        self.z = z

    def __call__(self, q):
        return power(self.z * 3*spherical_jn(1, q*self.r1) / (q*self.r1) * exp((-(q*self.s)**2)/2), 2)




class AtomicPlusNuclearFF:
    """
    combined electron cloud FF (Tsai parameterization) + Helm nuclear FF
    for Primakoff scattering at high energies
    """
    def __init__(self, n, z):
        self.z = z
        self.n = n
        self.s = 0.9 * (10 ** -15) / METER_BY_MEV
        self.r1 = sqrt((1.23*power(n+z, 1/3) - 0.6)**2 - 5*0.9**2 + 7*power(pi*0.52, 2)/3) * (10 ** -15) / METER_BY_MEV

    def __call__(self, q):
        t = q**2
        a = 184.15*np.power(2.718, -1/2)*np.power(self.z, -1/3) / M_E
        ff_a = abs(self.z*(t*a**2) / (1 + t*a**2))
        ff_helm = abs(self.z * 3*spherical_jn(1, q*self.r1) / (q*self.r1) * exp((-(q*self.s)**2)/2))
        return np.power(ff_a - self.z + ff_helm, 2)



class ProtonFF:
    """
    Square of the proton form factor F1
    """
    def __init__(self):
        pass

    def __call__(self, t):
        g_e = power(1 - t/0.71e6, -2)
        return power((g_e - t/(4*M_P**2))/(1 - t/(4*M_P**2)), 2)




def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator