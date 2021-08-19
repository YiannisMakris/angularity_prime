from numpy import pi, log, exp, real, absolute
from special_functions import *
from param import *
from renormalization_group import *
import cmath

def set_kernels(a_s, kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR, log_accuracy = log_accuracy):

    def R(m,m_F):
        return a_s(m) / a_s (m_F)

    # The convolution Kernel - \eta (from Eq.(2.18))
    # ----------------------------------------------

    def etaN0LL(m,m_F,Gq):
        r = R(m,m_F)
        return - Gq[0] / ( 2 * b[0] ) * log(r)

    def etaN1LL(m,m_F,Gq):
        r = R(m,m_F)
        return - Gq[0] / ( 2 * b[0] ) * a_s(m_F) / (4 * pi) * (r - 1) *           \
        (Gq[1]/ Gq[0] - b[1] / b[0] )
        
    def etaN2LL(m,m_F,Gq):
        r = R(m,m_F)
        return - Gq[0] / ( 4 * b[0] ) * (a_s(m_F) / (4 * pi))**2 * (r**2 - 1)     \
        * (B[2] + (Gq[2] / Gq[0]) - Gq[1] * b[1] / (Gq[0] * b[0])) 
        
    def etaN3LL(m,m_F,Gq):
        r = R(m,m_F)
        return - Gq[0] / ( 6 * b[0] ) * (a_s(m_F) / (4 * pi))**3 * (r**3 - 1)     \
        * (B[3] + (Gq[1] / Gq[0]) *  B[2] - Gq[2] * b[1] / (Gq[0] * b[0])         \
        + Gq[3] / Gq[0] ) 
        
    # The convolution Kernel - K_{\gamma} (from Eq.(2.20))
    # ---------------------------------------------------
        
    def kN0LL(m,m_F,Gq):
        return 0

    def kN1LL(m,m_F,Gq):
        r = R(m,m_F)
        return - Gq[0] / ( 2 * b[0] ) * log(r)

    def kN2LL(m,m_F,Gq):
        r = R(m,m_F)
        return - 1 / ( 2 * b[0] ) * a_s(m_F) / (4 * pi) * (r - 1) *               \
        (Gq[1] - Gq[0] * b[1] / b[0] )
        
    def kN3LL(m,m_F,Gq):
        r = R(m,m_F)
        return - 1 / ( 4 * b[0] ) * (a_s(m_F) / (4 * pi))**2 * (r**2 - 1)         \
        * (B[2] * Gq[0] + Gq[2]  - Gq[1] * b[1] /  b[0] ) 
        
        

    # The convolution Kernel - \tilde{K}_F (from Eq.(2.22))
    # ---------------------------------------------------
        
    def KN0LL(m,m_F,Q):
        r = R(m,m_F)
        rQ = R(m_F,Q)
        return Gq[0] / ( 4 * b[0]**2 ) * 4 * pi / a_s(m_F) * ( rQ * log(r) \
        + 1 / r - 1)
        
    def KN1LL(m,m_F,Q):
        r = R(m,m_F)
        rQ = R(m_F,Q)
        return Gq[0] / ( 4 * b[0]**2 ) * ( (Gq[1] / Gq[0] - b[1] / b[0]) * (      \
        rQ * (r - 1) -log(r) ) - b[1] / (2 * b[0]) * ((log(r))**2 +               \
        2 * log(r) * log(rQ) ) )
        
    def KN2LL(m,m_F,Q):
        r = R(m,m_F)
        rQ = R(m_F,Q)
        return Gq[0] / ( 4 * b[0]**2 ) * a_s(m_F) / (4 * pi ) * ( ( Gq[2]/Gq[0]   \
        -  b[1] * Gq[1] / (b[0] * Gq[0]) ) * ((r - 1)**2 / 2 + (r**2 - 1)         \
        * (rQ - 1) / 2 ) + B[2] * (rQ * (r**2 - 1) / 2 - log(r) / rQ) + (b[1] *   \
        Gq[1] / (b[0] * Gq[0]) - (b[1] / b[0])**2) * ((r - 1) * (1 - log(rQ)) - r \
          * log(r) ) )
        
    def KN3LL(m,m_F,Q):
        r = R(m,m_F)
        rQ = R(m_F,Q)
        return Gq[0] / ( 4 * b[0]**2 ) * (a_s(m_F) / (4 * pi ))**2 * ( ( Gq[3]    \
        / Gq[0] - Gq[2] * b[1] / (Gq[0] * b[0]) + B[2] * Gq[1] / Gq[0] +B[3]) *   \
        (r**3 -1) / 3 * rQ - b[1] / (2 * b[0]) * (Gq[2] / Gq[0] - Gq[1] * b[1]    \
        / (Gq[0] * b[0]) + B[2]) * (r**2 * log(r) +(r**2 - 1) * log(rQ) ) - B[3]  \
        * log(r) / (2 * r**2) + (b[3] / b[0] + b[1] * b[2] / b[0]**2 - 2 * Gq[3]  \
        / Gq[0] + 3 * Gq[2] * b[1] / (Gq[0] * b[0]) - Gq[1] * b[1]**2 / (Gq[0]    \
        * b[0]**2) ) * (r**2 - 1) / 4 + B[2] * (Gq[1]/Gq[0] - b[1]/b[0] * (1 - r) \
        /rQ) )

        
    # The G-Kernel - From Eq.(4.23) 
    # -----------------------------
        
    def G(a):
        return 2 * pi / b[0] * (1 / a + b[1] / (4 * pi * b[0]) * log(a) - B[2] /  \
        (4 * pi)**2 * a)  

# Adding to logarithmic accuracy
#-------------------------------

    if (log_accuracy == 0 ) :
        def eta(m,m_F,Gq):
            return etaN0LL(m,m_F,Gq) 
        
        def k(m,m_F,Gq):
            return kN0LL(m,m_F,Gq)
        
        def K(m,m_F,Q):
            return KN0LL(m,m_F,Q)
        
    elif (log_accuracy == 1 ):
        def eta(m,m_F,Gq):
            return etaN0LL(m,m_F,Gq) + etaN1LL(m,m_F,Gq)
        
        def k(m,m_F,Gq):
            return kN0LL(m,m_F,Gq) + kN1LL(m,m_F,Gq)
        
        def K(m,m_F,Q):
            return KN0LL(m,m_F,Q) + KN1LL(m,m_F,Q)
        
    elif (log_accuracy == 2 ):
        def eta(m,m_F,Gq):
            return etaN0LL(m,m_F,Gq) + etaN1LL(m,m_F,Gq) + etaN2LL(m,m_F,Gq)
        
        def k(m,m_F,Gq):
            return kN0LL(m,m_F,Gq) + kN1LL(m,m_F,Gq) + kN2LL(m,m_F,Gq)
        
        def K(m,m_F,Q):
            return KN0LL(m,m_F,Q) + KN1LL(m,m_F,Q) + KN2LL(m,m_F,Q)
        
    elif (log_accuracy == 3 ):
        def eta(m,m_F,Gq):
            return etaN0LL(m,m_F,Gq) + etaN1LL(m,m_F,Gq) + etaN2LL(m,m_F,Gq) +    \
            etaN3LL(m,m_F,Gq)
            
        def k(m,m_F,Gq):
            return kN0LL(m,m_F,Gq) + kN1LL(m,m_F,Gq) + kN2LL(m,m_F,Gq) +          \
            kN3LL(m,m_F,Gq)
            
        def K(m,m_F,Q):
            return KN0LL(m,m_F,Q) + KN1LL(m,m_F,Q) + KN2LL(m,m_F,Q) +             \
            KN3LL(m,m_F,Q)
        
    else:
        def eta(m,m_F,Gq):
            return etaN0LL(m,m_F,Gq) + etaN1LL(m,m_F,Gq) + etaN2LL(m,m_F,Gq) +    \
            etaN3LL(m,m_F,Gq)
            
        def k(m,m_F,Gq):
            return kN0LL(m,m_F,Gq) + kN1LL(m,m_F,Gq) + kN2LL(m,m_F,Gq) +          \
            kN3LL(m,m_F,Gq)
            
        def K(m,m_F,Q):
            return KN0LL(m,m_F,Q) + KN1LL(m,m_F,Q) + KN2LL(m,m_F,Q) +             \
            KN3LL(m,m_F,Q)
        
        print("WARNING: cross_section: beyond stable accuracy ")
        
    # Total kenrnel to logarithmic accuracy
    #--------------------------------------
        
    def Omega(m, mJ, mS):
        return - (2* kJ * eta(m, mJ, Gq) + kS * eta(m, mS, Gq))
        
    def TK(m, mH, mJ, mS, Q):
        return -(kH * K(m,mH,Q) + 2 * jJ * kJ * K(m,mJ,Q) + kS * K(m,mS,Q))

    def Tk(m, mH, mJ, mS):
        return k(m,mH,gH) + 2 * k(m,mJ,gJ) + k(m,mS,gS)

    # Renormalon Gap parameter
# ------------------------

    def Da(m, R, RStar, RD, initial, loc):
        r1 = b[1] / (2 * b[0]**2)
        r2 = 2 * pi / (b[0] * a_s(RStar )) 
        r3 = 2 * pi / (b[0] * a_s(RD))
        eta = etaN0LL(m,R,Gq) + etaN1LL(m,R,Gq)
        GD = G(a_s(RD))
        return real(initial/(1-avec[loc]) + RStar * exp(euler_gamma) * kS / 2 * eta  \
        + RD /(2 * b[0]) * exp(-GD) * (2 * pi / b[0] * exp(1j * pi))** r1 \
        * (-1 / (2 * b[0]) * (gR[1] - gR[0] / b[0] *(b[1] + B[2] / 2) ) \
        * D_Gammainc(-r1,-r2,-r3, 6)))

    return [Omega, TK, Tk, Da]
