# Detection Cross Sections
# All cross sections in cm^2
# All energies in MeV
from .constants import *
from .fmath import *

import multiprocessing as multi
from matplotlib.pyplot import hist2d


# Define ALP DETECTION cross-sections

#### Photon Coupling ####

def iprimakoff_dsigma_dtheta(theta, ea, g, ma, z, r0):
    # inverse-Primakoff scattering differential xs by theta
    # r0: screening parameter
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    q2 = -2*ea**2 + ma**2 + 2*ea*sqrt(ea**2 - ma**2)*cos(theta)
    beta = sqrt(ea**2 - ma**2)/ea
    return prefactor * (1 - exp(q2 * r0**2 / 4))**2 * (beta * sin(theta)**3)/(1+beta**2 - 2*beta*cos(theta))**2




def iprimakoff_nsigma(ea, g, ma, z, r0):
    # inverse-Primakoff scattering total xs (numerically integrated)
    # r0: screening parameter
    return quad(iprimakoff_dsigma_dtheta, 0, pi, args=(ea,g,ma,z,r0))[0]




def iprimakoff_sigma(ea, g, ma, z, r0):
    # inverse-Primakoff scattering total xs (Creswick et al)
    # r0: screening parameter
    prefactor = (g * z)**2 / (2*137)
    eta2 = r0**2 * (ea**2 - ma**2)
    return heaviside(ea-ma, 0.0)*prefactor * (((2*eta2 + 1)/(4*eta2))*log(1+4*eta2) - 1)




def dark_iprim_dsigma_dt(t, s, gZN, gaGZ, ma, mZp, M):
    # inverse Priamkoff with massive vector mediator (a + N -> \gamma + N via Z')
    prefactor = (gZN*gaGZ)**2 / (16*pi) / ((M + ma)**2 - s) / ((M - ma)**2 - s)
    return prefactor * (ma**2 * t * (M**2 + s) - (M*ma**2)**2 - t*((s-M**2)**2 + s*t) - t*(t-ma**2)/2) / (t-mZp**2)**2




def dark_iprim_dsigma_dcostheta(cosTheta, Ea, gZN, gaGZ, ma, mZp, z=6):
    # inverse Priamkoff with massive vector mediator (a + N -> \gamma + N via Z')
    prefactor = sqrt(M_P*(Ea - ma)*(2*Ea*M_P + ma**2))/(4*sqrt(2)*pi**2 * (2*Ea*M_P + M_P**2 + ma**2))
    t = ma**2 - ((ma**2 + 2*Ea*M_P)/(M_P + Ea - sqrt(Ea**2 - ma**2)*cosTheta)) * (Ea - sqrt(Ea**2 - ma**2)*cosTheta)
    s = M_P**2 + ma**2 + 2*Ea*M_P
    return prefactor * dark_iprim_dsigma_dt(t, s, gZN, gaGZ, ma, mZp, 2*z*M_P)




#### Electron Coupling ####

def axioelectric_xs(pe_xs, energy, z, a, g, ma):
    # Axio-electric total cross section for ionization
    pe = np.interp(energy, pe_xs[:,0], pe_xs[:,1])*1e-24 / (100*METER_BY_MEV)**2
    beta = sqrt(energy**2 - ma**2)
    return 137 * 3 * g**2 * pe * energy**2 * (1 - np.power(beta, 2/3)/3) / (16*pi*M_E**2 * beta)




def icompton_sigma_old(ea, g, z=1):
    # Total cross section (a + e- -> \gamma + e-)
    a = 1 / 137
    aa = g ** 2 / 4 / pi
    prefact = a * aa * pi / 2 / M_E / ea**2
    sigma = prefact * 2 * ea * (-(2*ea * (3*ea + M_E)/(2 * ea + M_E)**2) + np.log(2 * ea / M_E + 1))
    return z**2 * sigma




def icompton_sigma(ea, ma, g, z=1):
    # Borexino 2008, eq. 14
    y = 2 * M_E * ea + ma**2
    pa = sqrt((ea**2 - ma**2)*heaviside(ea - ma, 0.0))
    prefactor = (z**2) * ALPHA * power(g/M_E, 2) / (8 * pa)

    return prefactor * ((2 * M_E**2 * (M_E + ea) * y)/power(M_E**2 + y, 2) \
        + (4*M_E*(ma**4 + 2*power(ma*M_E, 2) - power(2*M_E*ea, 2)))/(y*(M_E**2 + y)) \
        + log((M_E + ea + pa)/(M_E + ea - pa))*(power(2*M_E*pa, 2) + ma**4)/(ea*y))





def icompton_dsigma_det(ea, et, g, ma):
    # Differential cross section by electron recoil (a + e- -> \gamma + e-)
    # dSigma / dEt   electron kinetic energy
    # ea: axion energy
    # et: transferred electron energy = E_e - m_e.
    y = 2 * M_E * ea + ma ** 2
    prefact = (1/137) * g ** 2 / (4 * M_E ** 2)
    pa = np.sqrt(ea ** 2 - ma ** 2)
    eg = ea - et
    return -(prefact / pa) * (1 - (8 * M_E * eg / y) + (12 * (M_E * eg / y) ** 2)
                                - (32 * M_E * (pa * ma) ** 2) * eg / (3 * y ** 3))




def icompton_dsigma_domega(theta, Ea, ma, ge):
    # Compton cross section dSigmadOmega
    y = 2*M_E*Ea + ma**2
    pa = sqrt(Ea**2 - ma**2)
    e_gamma = 0.5*y/(M_E + Ea - pa*cos(theta))

    prefactor = ge**2 * ALPHA * e_gamma / (4*pi*2*pa*M_E**2)
    return prefactor * (1 + 4*(M_E*e_gamma/y)**2 - 4*M_E*e_gamma/y - 4*M_E*e_gamma*(ma*pa*sin(theta))**2 / y**3)




# Define form factors

def _nuclear_ff(t, m, z, a):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    # a: number of nucleons
    return (2*m*z**2) / (1 + t / 164000*np.power(a, -2/3))**2



def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 184*np.power(2.718, -1/2)*np.power(z, -1/3) / M_E
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 1194*np.power(2.718, -1/2)*np.power(z, -2/3) / M_E
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator