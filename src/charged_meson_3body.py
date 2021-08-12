# Classes and constants for axion production and detection from 3-body decay of charged mesons

from .constants import *
from .fmath import *

# Proton total cross section
def sigmap(p):
    A = 307.8
    B = 0.897
    C = -2.598
    D = -4.973
    n = 0.003
    return A + B*power(p,n) + C*log(p)*log(p) + D*log(p)

sigmap_total = sigmap(7.944)  # GeV^-2

# pi+ decay probability
def pi_decay(p_pi):
    e_pi = sqrt(p_pi**2 + M_PI**2)
    boost = e_pi / M_PI
    v_pi = p_pi / e_pi
    prob = exp(-50 / (METER_BY_MEV*v_pi*boost*2.6e-8 / HBAR))
    return (1 - prob)



# Charged pion production double-differential cross section on Be target
def meson_production_d2SdpdOmega(p, theta, p_proton, meson_type="pi_plus"):
    pB = p_proton
    mt = M_P
    # Sanford-Wang Parameterization
    if meson_type == "pi_plus":
        c1 = 220.7
        c2 = 1.080
        c3 = 1.0
        c4 = 1.978
        c5 = 1.32
        c6 = 5.572
        c7 = 0.0868
        c8 = 9.686
        c9 = 1.0
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential
    elif meson_type == "pi_minus":
        c1 = 213.7
        c2 = 0.9379
        c3 = 5.454
        c4 = 1.210
        c5 = 1.284
        c6 = 4.781
        c7 = 0.07338
        c8 = 8.329
        c9 = 1.0
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential
    elif meson_type == "k_plus":
        pT = p*sin(theta)
        pL = p*cos(theta)
        beta = pB / (mt*1e-3 + sqrt(pB**2 + (M_P*1e-3)**2))
        gamma = power(1-beta**2, -0.5)
        pLstar = gamma*(pL - sqrt(pL**2 + pT**2 + (M_K*1e-3)**2)*beta)
        s = (M_P*1e-3)**2 + (mt*1e-3)**2 + 2*sqrt(pB**2 + (M_P*1e-3)**2)*mt*1e-3
        xF = abs(2*pLstar/sqrt(s))

        c1 = 11.70
        c2 = 0.88
        c3 = 4.77
        c4 = 1.51
        c5 = 2.21
        c6 = 2.17
        c7 = 1.51
        prefactor = c1 * p**2 / sqrt(p**2 + (M_K*1e-3)**2)
        return prefactor * (1 - xF) * exp(-c2*pT - c3*power(xF, c4) - c5*pT**2 - c7*power(pT*xF, c6))
    elif meson_type == "k0S":
        c1 = 15.130
        c2 = 1.975
        c3 = 4.084
        c4 = 0.928
        c5 = 0.731
        c6 = 4.362
        c7 = 0.048
        c8 = 13.300
        c9 = 1.278
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential



# Compton cross section dSigmadOmega
def compton_dSigmadOmega(theta, Ea, ma, ge):
    y = 2*M_E*Ea + ma**2
    pa = sqrt(Ea**2 - ma**2)
    e_gamma = 0.5*y/(M_E + Ea - pa*cos(theta))

    prefactor = ge**2 * ALPHA * e_gamma / (4*pi*2*pa*M_E**2)
    return prefactor * (1 + 4*(M_E*e_gamma/y)**2 - 4*M_E*e_gamma/y - 4*M_E*e_gamma*(ma*pa*sin(theta))**2 / y**3)


# Dark Primakoff cross section
def dark_prim_dSdt(t, s, gZN, gaGZ, ma, mZp, M):
    # Priamkoff with massive vector mediator
    prefactor = (gZN*gaGZ)**2 / (16*pi) / ((M + ma)**2 - s) / ((M - ma)**2 - s)
    return prefactor * (ma**2 * t * (M**2 + s) - (M*ma**2)**2 - t*((s-M**2)**2 + s*t) - t*(t-ma**2)/2) / (t-mZp**2)**2

def dark_prim_dSdCosTheta(cosTheta, Ea, gZN, gaGZ, ma, mZp, z=6):
    prefactor = sqrt(M_P*(Ea - ma)*(2*Ea*M_P + ma**2))/(4*sqrt(2)*pi**2 * (2*Ea*M_P + M_P**2 + ma**2))
    t = ma**2 - ((ma**2 + 2*Ea*M_P)/(M_P + Ea - sqrt(Ea**2 - ma**2)*cosTheta)) * (Ea - sqrt(Ea**2 - ma**2)*cosTheta)
    s = M_P**2 + ma**2 + 2*Ea*M_P
    return prefactor * dark_prim_dSdt(t, s, gZN, gaGZ, ma, mZp, 2*z*M_P)




class ChargedPionFluxMiniBooNE:
    def __init__(self, proton_energy=8000.0):
        self.n_samples = 10000
        self.ep = proton_energy
        self.x0 = np.array([])
        self.y0 = np.array([])
        self.z0 = np.array([])
        self.px0 = np.array([])
        self.py0 = np.array([])
        self.pz0 = np.array([])

    def sigmap(self, p):
        A = 307.8
        B = 0.897
        C = -2.598
        D = -4.973
        n = 0.003
        return A + B*power(p,n) + C*log(p)*log(p) + D*log(p)

    def d2SdpdOmega_SW(self):
        pass

    def simulate_beam_spot(self):
        r1 = norm.rvs(size=self.n_samples)
        r2 = norm.rvs(size=self.n_samples)
        r3 = norm.rvs(size=self.n_samples)
        r4 = norm.rvs(size=self.n_samples)

        sigma_x = 1.51e-1  # cm
        sigma_y = 0.75e-1  # cm
        sigma_theta_x = 0.66e-3  # mrad
        sigma_theta_y = 0.40e-3  # mrad

        self.x0 = r1*sigma_x
        self.y0 = r2*sigma_y
        self.z0 = -10.0

        self.px0 = sqrt(self.ep**2 - M_P**2)*r3*sigma_theta_x
        self.py0 = sqrt(self.ep**2 - M_P**2)*r4*sigma_theta_y
        self.pz0 = sqrt(self.ep**2 - M_P**2 - self.px0**2 - self.py0**2)

    def B(self, r):
        # B field in T for r in cm
        return heaviside(r - 2.2, 0.0) * (4*pi*1e-2) * 170 / (2*pi*r)

    def focus_pions(self):
        pass




# 1206.3587 efficiency for mu detection
eff_data = np.genfromtxt("data/3body/efficiency_1206-3587.txt", delimiter=",")
def Dmu(Emu):
    return np.interp(Emu, eff_data[:,0], eff_data[:,1])

cp_data = np.genfromtxt("data/3body/scalar_coupling_1206-3587.txt", delimiter=",")
cs_data = np.genfromtxt("data/3body/pseudoscalar_coupling_1206-3587.txt", delimiter=",")
cv_data = np.genfromtxt("data/3body/vector_coupling_1206-3587.txt", delimiter=",")
def cp(m):
    return sqrt(4*pi*np.interp(m, cp_data[:,0], cp_data[:,1]))

def cs(m):
    return sqrt(4*pi*np.interp(m, cs_data[:,0], cs_data[:,1]))

def cv(m):
    return sqrt(4*pi*np.interp(m, cv_data[:,0], cv_data[:,1]))

# Convolve flux with axion branching ratio and generate ALP flux
class ChargedMeson3BodyDecay:
    def __init__(self, pion_flux, axion_mass=0.1, coupling=1.0, n_samples=50,
                 meson_mass=M_PI, ckm=V_UD, fM=F_PI, boson_type="P"):
        self.pion_flux = pion_flux
        self.mm = meson_mass
        self.ckm = ckm
        self.fM = fM
        self.rep = boson_type
        self.ma = axion_mass
        self.gmu = coupling
        self.det_dist = 541
        self.dump_dist = 50
        self.det_length = 12
        self.det_sa = cos(arctan(self.det_length/(self.det_dist-self.dump_dist)/2))
        self.nsamples = n_samples
        self.energies = []
        self.cosines = []
        self.weights = []
        self.decay_weight = []
        self.scatter_weight = []
    
    def lifetime(self, ge):
        if 1 < 4 * (M_E / self.ma) ** 2:
            return np.inf
        return (8 * pi) / (ge**2 * self.ma * power(1 - 4 * (M_E / self.ma) ** 2, 1 / 2))

    def dGammadEa(self, Ea):
        m212 = self.mm**2 + self.ma**2 - 2*self.mm*Ea
        e2star = (m212 - M_MU**2)/(2*sqrt(m212))
        e3star = (self.mm**2 - m212 - self.ma**2)/(2*sqrt(m212))

        if self.ma > e3star:
            return 0.0

        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2) - sqrt(e3star**2 - self.ma**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2) + sqrt(e3star**2 - self.ma**2))**2
    
        def MatrixElement2P(m223):
            ev = (m212 + m223 - M_MU**2 - self.ma**2)/(2*self.mm)
            emu = (self.mm**2 - m223 + M_MU**2)/(2*self.mm)
            q2 = self.mm**2 - 2*self.mm*ev

            prefactor = heaviside(e3star-self.ma,0.0)*(self.gmu*G_F*self.fM*self.ckm/(q2 - M_MU**2))**2
            return Dmu(emu)*prefactor*((2*self.mm*emu*q2 * (q2 - M_MU**2) - (q2**2 - (M_MU*self.mm)**2)*(q2 + M_MU**2 - self.ma**2)) - (2*q2*M_MU**2 * (self.mm**2 - q2)))
        
        def MatrixElement2S(m223):
            ev = (m212 + m223 - M_MU**2 - self.ma**2)/(2*self.mm)
            emu = (self.mm**2 - m223 + M_MU**2)/(2*self.mm)
            q2 = self.mm**2 - 2*self.mm*ev

            prefactor = heaviside(e3star-self.ma,0.0)*(self.gmu*G_F*self.fM*self.ckm/(q2 - M_MU**2))**2
            return Dmu(emu)*prefactor*((2*self.mm*emu*q2 * (q2 - M_MU**2) - (q2**2 - (M_MU*self.mm)**2)*(q2 + M_MU**2 - self.ma**2)) + (2*q2*M_MU**2 * (self.mm**2 - q2)))

        def MatrixElement2V(m223):
            pass
        
        if self.rep == "P":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2P, m223Min, m223Max)[0]
        
        if self.rep == "S":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2S, m223Min, m223Max)[0]

        if self.rep == "V":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2V, m223Min, m223Max)[0]
    
    def dGammadEmuV(self, Emu):
        def dGammadEnudEmuV(Enu):
            cr = self.gmu
            cl = self.gmu
            q2 = self.mm**2 - 2*self.mm*Enu
            EV = self.mm - Enu - Emu
            prefactor = self.gamma_sm() * self.mm**2 / power(2*pi*M_MU*(self.mm*2 - M_MU**2)*(q2 - M_MU**2), 2)
            return prefactor * (4*power(cr*M_MU*self.mm, 2)*Emu*Enu - 12*cr*cl*M_MU**2 * self.mm*q2*Enu \
                + (power(cl*q2, 2) - power(cr*M_MU*self.mm, 2))*(self.mm**2 + self.ma**2 - M_MU**2 - 2*self.mm*EV) \
                    + power(self.ma, -2)*(self.mm**2 - self.ma**2 - M_MU**2 - 2*self.mm*Enu)*(4*power(cr*M_MU*self.mm, 2)*EV*Enu \
                        + (power(cl*q2, 2) - power(cr*M_MU*self.mm, 2))*(self.mm**2 - self.ma**2 + M_MU**2 - 2*self.mm*Emu)))
        
        if Emu < M_MU:
            return 0.0
        Enu_max = (self.mm**2 + M_MU**2 - self.ma**2 - 2*self.mm*Emu)/(2*(self.mm - Emu - sqrt(Emu**2 - M_MU**2)))
        Enu_min = (self.mm**2 + M_MU**2 - self.ma**2 - 2*self.mm*Emu)/(2*(self.mm - Emu + sqrt(Emu**2 - M_MU**2)))
        return Dmu(Emu)*quad(dGammadEnudEmuV, Enu_min, Enu_max)[0]
    
    def BrV(self):
        def dGammadEnudEmuV(Enu, Emu):
            cr = 0.0
            cl = self.gmu
            q2 = self.mm**2 - 2*self.mm*Enu
            EV = self.mm - Enu - Emu
            prefactor = heaviside(Emu-M_MU,0.0)*self.mm**2 / power(2*pi*M_MU*(self.mm*2 - M_MU**2)*(q2 - M_MU**2), 2)
            return Dmu(Emu)*prefactor * (4*power(cr*M_MU*self.mm, 2)*Emu*Enu - 12*cr*cl*M_MU**2 * self.mm*q2*Enu \
                + (power(cl*q2, 2) - power(cr*M_MU*self.mm, 2))*(self.mm**2 + self.ma**2 - M_MU**2 - 2*self.mm*EV) \
                    + power(self.ma, -2)*(self.mm**2 - self.ma**2 - M_MU**2 - 2*self.mm*Enu)*(4*power(cr*M_MU*self.mm, 2)*EV*Enu \
                        + (power(cl*q2, 2) - power(cr*M_MU*self.mm, 2))*(self.mm**2 - self.ma**2 + M_MU**2 - 2*self.mm*Emu)))
        
        def Enu_max(Emu): 
            return (self.mm**2 + M_MU**2 - self.ma**2 - 2*self.mm*Emu)/(2*(self.mm - Emu - sqrt(Emu**2 - M_MU**2)))
        def Enu_min(Emu):
            return (self.mm**2 + M_MU**2 - self.ma**2 - 2*self.mm*Emu)/(2*(self.mm - Emu + sqrt(Emu**2 - M_MU**2)))
        return dblquad(dGammadEnudEmuV, M_MU, (self.mm**2 + M_MU**2 - self.ma**2)/(2*self.mm), Enu_min, Enu_max)[0]
    
    def total_br_V(self):
        return quad(self.dGammadEmuV, M_MU, (self.mm**2 + M_MU**2 - self.ma**2)/(2*self.mm))[0] / self.gamma_sm()

    def gamma_sm(self):
        return (G_F*self.fM*M_MU*self.ckm)**2 * self.mm * (1-(M_MU/self.mm)**2)**2 / (4*pi)

    def total_br(self):
        EaMax = (self.mm**2 + self.ma**2 - M_MU**2)/(2*self.mm)
        EaMin = self.ma
        return quad(self.dGammadEa, EaMin, EaMax)[0] / self.gamma_sm()
    
    def simulate_single(self, pion_p, pion_theta, pion_wgt):
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - M_MU**2)/(2*self.mm)

        # Draw random variate energies and angles in the pion rest frame
        energies = np.random.uniform(ea_min, ea_max, self.nsamples)
        momenta = sqrt(energies**2 - self.ma**2)
        cosines = np.random.uniform(-1, 1, self.nsamples)
        pz = momenta*cosines

        weights = np.array([pion_wgt*(ea_max - ea_min)*self.dGammadEa(ea) / self.gamma_sm() / self.nsamples for ea in energies])

        # Boost to lab frame
        beta = pion_p / sqrt(pion_p**2 + self.mm**2)
        boost = power(1-beta**2, -0.5)
        e_lab = boost*(energies - beta*pz)
        pz_lab = boost*(pz - beta*energies)
        cos_theta_lab = -pz_lab / sqrt(e_lab**2 - self.ma**2)

        for i in range(self.nsamples):
            solid_angle_acceptance = heaviside(cos_theta_lab[i] - self.det_sa, 0.0)
            if solid_angle_acceptance == 0.0:
                continue
            self.energies.append(e_lab[i])
            self.cosines.append(cos_theta_lab[i])
            self.weights.append(2*pi*weights[i]*heaviside(e_lab[i]-140.0,1.0))
    
    def simulate(self):
        self.energies = []
        self.cosines = []
        self.weights = []
        self.scatter_weight = []
        self.decay_weight = []
        for i, p in enumerate(self.pion_flux):
            self.simulate_single(p[0], p[1], p[2])
        

    def propagate(self, ge):  # propagate to detector
        e_a = np.array(self.energies)
        wgt = np.array(self.weights)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.ma**2)
        v_a = p_a / e_a
        axion_boost = e_a / self.ma

        surv_prob = exp(-self.det_dist / METER_BY_MEV / v_a / (axion_boost * self.lifetime(ge)))
        decay_prob = 1.0 - exp(-self.det_length / METER_BY_MEV / v_a / (axion_boost * self.lifetime(ge)))
        
        self.decay_weight = np.asarray(wgt * surv_prob * decay_prob, dtype=np.float64)
        self.scatter_weight = np.asarray(wgt * surv_prob, dtype=np.float64)
    
    def scatter_compton(self, ge, n_e, cosine_bins):
        # make a histogram
        h = np.histogram([0.0], weights=[0.0], bins=cosine_bins)[0]
        centers = (cosine_bins[1:] + cosine_bins[:-1])/2
        for i in range(self.scatter_weight.shape[0]):
            rcos = np.random.uniform(-1, 1, self.nsamples)
            rthetas = arccos(rcos)
            wgts = 4*pi*compton_dSigmadOmega(rthetas, self.energies[i], self.ma, ge)/self.nsamples
            h += np.histogram(rcos, weights=self.scatter_weight[i]*n_e*power(METER_BY_MEV*100, 2)*wgts, bins=cosine_bins)[0]
        return h, centers
    
    def scatter_dark_primakoff(self, gZN, gaGZ, mZp, n_e, cosine_bins):
        # make a histogram
        h = np.histogram([0.0], weights=[0.0], bins=cosine_bins)[0]
        centers = (cosine_bins[1:] + cosine_bins[:-1])/2
        for i in range(self.scatter_weight.shape[0]):
            rcos = np.random.uniform(-1, 1, self.nsamples)
            wgts = 4*pi*dark_prim_dSdCosTheta(rcos, self.energies[i], gZN, gaGZ, self.ma, mZp)/self.nsamples
            wgts = wgts * 4.75  # ad hoc coherency factor
            h += np.histogram(rcos, weights=self.scatter_weight[i]*n_e*power(METER_BY_MEV*100, 2)*wgts, bins=cosine_bins)[0] 
        return h, centers
