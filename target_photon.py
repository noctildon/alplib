import os
import pkg_resources
import numpy as np
from .materials import Material


class PhotonCoherent:
    """
    Get coherent photon nucleus cross section from https://physics.nist.gov/cgi-bin/Xcom/xcom3_1
    """
    def __init__(self, target: Material):
        self.material = target
        self.symbol = target.mat_name
        self.data = None

        folder_path = pkg_resources.resource_filename(__name__, 'data/photon_coherent')
        files = os.listdir(folder_path) # ['Th.txt']
        files = ['.'.join(x.split('.')[:-1]) for x in files] # ['Th']

        if self.symbol in files:
            # photon energy [MeV], cross section cm^2/g
            fpath = os.path.join(folder_path, self.symbol + '.txt')
            self.data = np.loadtxt(fpath)

            mass_number = self.material.z[0] + self.material.n[0]
            factor = 6e23 / mass_number
            self.data[:, 1] /= factor # 1/g -> 1/nucleus

    def xsec(self, energy):
        """
        energy: photon energy [MeV]
        return: cross section per nucleus [cm^2]
        """
        if self.data is not None:
            return np.interp(energy, self.data[:, 0], self.data[:, 1])
        return None