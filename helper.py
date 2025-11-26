import numpy as np

# ma in eV
def DFSZ(ma):
    return (0.203*8/3 - 0.39)*ma*1e-9

def DFSZII(ma):
    return (0.203*2/3 - 0.39)*ma*1e-9

def KSVZ(ma, eByN):
    return (0.203*eByN - 0.39)*ma*1e-9

def gaussian(x, mu, sigma):
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)