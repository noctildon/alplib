"""
Matrix element class
"""

from .fmath import *
from .constants import *
from .form_factors import *



class MatrixElement2:
    def __init__(self, m1, m2, m3, m4):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4

    def __call__(self, s, t):
        return 0.0




class MatrixElementDecay2:
    def __init__(self, m_parent, m1, m2):
        self.m_parent = m_parent
        self.m1 = m1
        self.m2 = m2

    def __call__(self):
        return 0.0




class M2VectorDecayToFermions(MatrixElementDecay2):
    def __init__(self, m_parent, m):
        super().__init__(m_parent, m, m)

    def __call__(self, coupling=1.0):
        return 4*(coupling**2)*(self.m_parent**2 - 2*self.m1**2)



class M2Chi2ToChi1Vector(MatrixElementDecay2):
    def __init__(self, m_chi2, m_chi1, m_v):
        super().__init__(m_chi2, m_chi1, m_v)
        # m_parent = m_chi2
        # m1 = m_chi1
        # m2 = m_v

    def __call__(self, coupling=1.0):
        return coupling**2 * (12*self.m1*self.m_parent - 2*self.m_parent**2 \
            - 2*power(self.m_parent/self.m2, 2) * (self.m_parent**2 - self.m1**2 - self.m2**2))



class M2DMUpscatter(MatrixElement2):
    """
    Dark matter upscattering (chi1 + N -> chi2 + N) via heavy mediator V
    """
    def __init__(self, mchi1, mchi2, mV, mN):
        super().__init__(mchi1, mN, mchi2, mN)
        self.mV = mV
        self.mchi1 = mchi1
        self.mchi2 = mchi2
        self.mN = mN
        self.ff = ProtonFF()

    def __call__(self, s, t, coupling_product=1.0):
        prefactor = ALPHA * self.ff(t) * coupling_product**2
        propagator = power(t - self.mV**2, 2)
        numerator = 8*(2*power(self.mN,4) + 4*self.mN**2 * (self.mchi1 * self.mchi2 - s) \
            + 2*(self.mchi1**2 - s)*(self.mchi2**2-s) - t*(self.mchi1-self.mchi2)**2 + 2*s*t + t**2)
        return prefactor * numerator / propagator




class M2DarkPrimakoff(MatrixElement2):
    """
    Dark Primakoff scattering (a + N -> gamma + N) via heavy mediator Zprime
    """
    def __init__(self, ma, mN, mZp):
        super().__init__(ma, mN, 0, mN)
        self.mZp = mZp
        self.mN = mN
        self.ma = ma

    def __call__(self, s, t, coupling_product=1.0):
        prefactor = ALPHA * coupling_product**2
        propagator = power(t - self.mZp**2, 2)
        numerator = (2*self.mN**2 * (self.ma**2 - 2*s - t) + 2*self.mN**4 - 2*self.ma**2 * (s + t) + self.ma**4 + 2*s**2 + 2*s*t + t**2)
        return prefactor * numerator / propagator




class M2PairProduction:
    """
    a + N -> e+ e- N    ALP-driven pair production
    """
    def __init__(self, ma, mN, n, z, ml=M_E):
        self.ma = ma
        self.mN = mN
        self.ff2 = AtomicPlusNuclearFF(n, z)
        self.ml = ml

    def sub_elements(self, kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case="alp"):
        if case == "alp":
            m1_2 = -32 * ( (M_E**2 - kp1)*(2*kl2*p2l1 + 2*kl1*p2l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
            m2_2 = -32 * ( (M_E**2 - kp2)*(2*kl2*p1l1 + 2*kl1*p1l2) + (self.ma**2 - 2*M_E**2)*(p2l1*p1l2 + p1l1*p2l2) )
            m2_m1 = -32 * ( kp1 * (kl2*p2l1 + kl1*p2l2 - power(M_E*self.mN, 2)) \
                            + kp2 * (kl2*p1l1 + kl1*p1l2 - power(M_E*self.mN, 2)) \
                            - 2*kl1*kl2*p1p2 - p2l1*p1l2*self.ma**2 + (M_E**2 - self.ma**2)*p1l1*p2l2 \
                            + p2l1*p1l2*M_E**2 + p1p2*power(M_E*self.ma,2) + power(self.mN*M_E**2, 2) )
            return m1_2, m2_2, m2_m1
        elif case == "vector":
            return 0.0
        elif case == "sm":
            m1_2 = -128.0 * ( kl2*p2l1*(M_E**2 - kp1) + p2l2*(kl1*(M_E**2 - kp1) - M_E**2 * p1l1) - M_E**2 * p1l2*p2l1 )
            m2_2 = -128.0 * ( kl2*p1l1*(M_E**2 - kp2) + p1l2*(kl1*(M_E**2 - kp2) - M_E**2 * p2l1) - M_E**2 * p2l2*p1l1 )
            m2_m1 = -64.0 * ( -kp1*(p2l1*(p1l2 - 2*p2l2) + p1l1*p2l2 + (M_E*self.mN)**2) \
                            - kp2*(p1l1*(p2l2 - 2*p1l2) + p2l1*p1l2 + (M_E*self.mN)**2) \
                            + p2l2*(M_E**2 * (kl1 + p1l1 - 2*p2l1) - p1p2*(kl1 - 2*p1l1)) \
                            + M_E**2 * (kl2*p1l1 + kl1*p1l2 + kl2*p2l1 + p2l1*p1l2 - 2*p1l1*p1l2) \
                            - kl2*p1l1*p1p2 - kl2*p2l1*p1p2 - kl1*p1p2*p1l2 + 2*p2l1*p1p2*p1l2 \
                            - power(M_E*self.mN, 2)*p1p2 + power(M_E, 4)*power(self.mN, 2))
            return m1_2, m2_2, m2_m1
        else:
            print("case=", case, " not found in M2PairProduction.")
            raise Exception()


    def m2(self, Ea, Ep, tp, tm, phi, coupling=1.0, case="alp"):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

        p1 = sqrt(Ep**2 - M_E**2)
        Em = Ea - Ep
        p2 = sqrt(Em**2 - M_E**2)
        k = sqrt(Ea**2 - self.ma**2)

        # 3-vector dot products
        l2_dot_k = k**2 - k*p1*c1 - k*p2*c2
        l2_dot_p1 = k*p1*c1 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        l2_dot_p2 = k*p2*c2 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        p1_dot_p2 = p1*p2*(s1*s2*cphi + c1*c2)

        # 4-vector scalar products
        kp1 = Ea*Ep - k*p1*c1
        kp2 = Ea*Em - k*p2*c2
        kl1 = Ea*self.mN
        kl2 = Ea*self.mN - l2_dot_k
        p1p2 = Ep*Em - p1_dot_p2
        p1l1 = Ep*self.mN
        p2l1 = Em*self.mN
        p1l2 = Ep*self.mN - l2_dot_p1
        p2l2 = Em*self.mN - l2_dot_p2

        m1_2, m2_2, m2_m1 = self.sub_elements(kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case)

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2

        propagator1 = q2*(self.ma**2 - 2*kp1)
        propagator2 = q2*(self.ma**2 - 2*kp2)

        prefactor = power(4*pi*ALPHA*coupling, 2) * self.ff2(sqrt(abs(q2)))

        return prefactor * (m1_2 / power(propagator1, 2) \
                            + m2_2 / power(propagator2, 2) \
                            + 2 * m2_m1 / propagator2 / propagator1)

    def m2_separated(self, Ea, Ep, tp, tm, phi, coupling=1.0, case="alp"):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

        p1 = sqrt(Ep**2 - M_E**2)
        Em = Ea - Ep
        p2 = sqrt(Em**2 - M_E**2)
        k = sqrt(Ea**2 - self.ma**2)

        # 3-vector dot products
        l2_dot_k = self.ma**2 - k*p1*c1 - k*p2*c2
        l2_dot_p1 = k*p1*c1 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        l2_dot_p2 = k*p2*c2 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        p1_dot_p2 = p1*p2*(s1*s2*cphi + c1*c2)

        # 4-vector scalar products
        kp1 = Ea*Ep - k*p1*c1
        kp2 = Ea*Em - k*p2*c2
        kl1 = Ea*self.mN
        kl2 = Ea*self.mN - l2_dot_k
        p1p2 = Ep*Em - p1_dot_p2
        p1l1 = Ep*self.mN
        p2l1 = Em*self.mN
        p1l2 = Ep*self.mN - l2_dot_p1
        p2l2 = Em*self.mN - l2_dot_p2

        m1_2, m2_2, m2_m1 = self.sub_elements(kp1, kp2, kl1, kl2, p1p2, p1l1, p2l1, p1l2, p2l2, case)

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2

        propagator1 = q2*(self.ma**2 - 2*kp1)
        propagator2 = q2*(self.ma**2 - 2*kp2)

        prefactor = power(4*pi*ALPHA*coupling, 2) * self.ff2(sqrt(abs(q2)))

        return prefactor * m1_2 / power(propagator1, 2), \
                prefactor * m2_2 / power(propagator2, 2), \
                prefactor * 2 * m2_m1 / (propagator2*propagator1)

    def m2_v2(self, Ea, Ep, tp, tm, phi, coupling=1.0):
        # k: ALP momentum
        # p1: positron momentum
        # p2: electron momentum
        # l1: initial nucleus momentum
        # l2: final nucleus momentum
        c1 = cos(tp)
        c2 = cos(tm)
        s1 = sin(tp)
        s2 = sin(tm)
        cphi = cos(phi)

        p1 = sqrt(Ep**2 - M_E**2)
        Em = Ea - Ep
        p2 = sqrt(Em**2 - M_E**2)
        k = sqrt(Ea**2 - self.ma**2)

        # 3-vector dot products
        l2_dot_k = self.ma**2 - k*p1*c1 - k*p2*c2
        l2_dot_p1 = k*p1*c1 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        l2_dot_p2 = k*p2*c2 - M_E**2 - p1*p2*(s1*s2*cphi + c1*c2)
        p1_dot_p2 = p1*p2*(s1*s2*cphi + c1*c2)

        # 4-vector scalar products
        kl1 = Ea*Ep - k*p1*c1
        kl2 = Ea*Em - k*p2*c2
        kp1 = Ea*self.mN
        kp2 = Ea*self.mN - l2_dot_k
        l1l2 = Ep*Em - p1_dot_p2
        l1p1 = Ep*self.mN
        l2p1 = Em*self.mN
        l1p2 = Ep*self.mN - l2_dot_p1
        l2p2 = Em*self.mN - l2_dot_p2
        p1p2 = self.mN**2

        q2 = self.ma**2 + 2*M_E**2 - 2*kp1 - 2*kp2 + 2*p1p2
        prefactor = power(4*pi*ALPHA*coupling / q2, 2) * self.ff2(sqrt(abs(q2)))
        lh = (1/((self.ma**2-2*kl1)**2*(self.ma**2-2*kl2)**2))*16*((2*((p1p2-3*self.mN**2)*M_E**2+2*l1p1*(l2p1+l2p2)-l1l2*(self.mN**2+p1p2))*self.ma**2+Ea*self.mN*(2*(self.mN**2-p1p2)*M_E**2+l1p1*(l2p2-l2p1))+kp2*(2*(self.mN**2-p1p2)*M_E**2+l1p1*(l2p1-l2p2)+4*Ea*(M_E**2+l1l2)*self.mN))*self.ma**4-2*kl2*((2*(p1p2-3*self.mN**2)*M_E**2+l1p1*(self.mN**2+4*l2p1+4*l2p2-p1p2)-2*l1l2*(self.mN**2+p1p2))*self.ma**2+kp2*(-l2p1*M_E**2-p1p2*M_E**2+self.mN*(4*Ea*(M_E**2+l1l2)-l1l2*self.mN)+l1p1*(M_E**2+2*self.ma**2+3*l2p1-l2p2))+Ea*self.mN*(-(-4*self.mN**2+l2p2+3*p1p2)*M_E**2+l1l2*self.mN**2+l1p1*(M_E**2+2*self.ma**2-2*l2p1)))*self.ma**2+4*kl2**3*(M_E**2-l1p1)*self.mN**2+4*kl1**3*(M_E**2+2*kl2-l2p2)*self.mN**2-2*kl1**2*(7*self.mN**2*self.ma**2*M_E**2+2*Ea*l1p1*self.mN*M_E**2-8*kl2**2*self.mN**2+2*l1l2*self.mN**2*self.ma**2-3*l2p2*self.mN**2*self.ma**2-2*l1p1*l2p1*self.ma**2-2*l1p1*l2p2*self.ma**2-2*Ea*l2p2*self.mN*self.ma**2-2*Ea*l1p1*l2p2*self.mN+(l2p2-3*M_E**2)*self.ma**2*p1p2+2*kl2*(2*kp2*(l1p1+l2p1)+self.mN*(2*Ea*(l1p1+l2p2)+self.mN*(M_E**2+4*self.ma**2+l2p2))-(l1p1+l2p2)*p1p2)+2*kp2*(p1p2*M_E**2-(2*M_E**2+l1l2)*self.mN**2-l2p1*self.ma**2+l1p1*(M_E**2+l2p1)))+2*kl2**2*((3*p1p2*M_E**2-(7*M_E**2+2*l1l2)*self.mN**2+l1p1*(3*self.mN**2+2*l2p1+2*l2p2-p1p2))*self.ma**2+2*kp2*(l1p1*(self.ma**2+l2p1)-l2p1*M_E**2)+2*Ea*self.mN*(-(-2*self.mN**2+l2p2+p1p2)*M_E**2+l1l2*self.mN**2+l1p1*(self.ma**2-l2p1)))+kl1*(8*self.mN**2*kl2**3-4*((M_E**2+4*self.ma**2+l1p1+l2p1-l2p2)*self.mN**2+2*Ea*(l1p1+l2p2)*self.mN+2*kp2*(l1p1+l2p1)-(l1p1+l2p1)*p1p2)*kl2**2+2*(((2*M_E**2+4*self.ma**2+2*l2p1-l2p2)*self.mN**2+l1p1*(self.mN**2+4*l2p1+4*l2p2-3*p1p2)-(2*M_E**2+4*l1l2+2*l2p1+l2p2)*p1p2)*self.ma**2+2*Ea*self.mN*(-p1p2*M_E**2-l1l2*self.mN**2+(l1p1+l2p2)*(M_E**2+3*self.ma**2))+2*kp2*(-p1p2*M_E**2+self.mN*(4*Ea*(M_E**2+l1l2)-l1l2*self.mN)+l2p1*(M_E**2+3*self.ma**2)+l1p1*(M_E**2+3*self.ma**2+l2p1-l2p2)))*kl2-self.ma**2*(((-12*M_E**2-4*l1l2+l2p1+l2p2)*self.mN**2+8*l1p1*(l2p1+l2p2)-(-4*M_E**2+4*l1l2+l2p1+l2p2)*p1p2)*self.ma**2+2*kp2*(-3*p1p2*M_E**2-l1p1*(M_E**2+l2p1+l2p2)+4*Ea*l1l2*self.mN+self.mN*(4*(Ea+self.mN)*M_E**2+l1l2*self.mN)+l2p1*(M_E**2+2*self.ma**2))-2*Ea*self.mN*(p1p2*M_E**2+l1l2*self.mN**2+l1p1*(M_E**2-2*l2p2)-l2p2*(M_E*2+2*self.ma**2)))))
        return prefactor * lh