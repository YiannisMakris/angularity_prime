
#************  Packages etc.  *****************
import numpy as np
from numpy import pi, log, exp
from numpy import euler_gamma 
from param import *
from scipy.special import zeta

#************  Beta function *****************
b = np.zeros(4)
b[0] = (11 / 3) * CA - (4 / 3) * TF * nf
b[1] = (34 / 3) * (CA**2) - ( (20 / 3) * CA + 4 * CF ) * TF * nf 
b[2] = (2857 / 54) * (CA**3) + ( CF**2 - (205 / 18) * CF * CA                 \
     - (1415 / 54) * CA**2) * 2 * TF * nf + ( (11 / 9) * CF                   \
     + (79 / 54) * CA ) * 4 * TF**2 * nf**2
b[3] = 4826.16  # For NC = 3 and nf = 5 

B = np.zeros(4)
B[2] = (b[1] / b[0])**2 - (b[2] / b[0])
B[3] = -(b[1] / b[0])**3 + 2 * (b[2] * b[1] / b[0]**2) - (b[3] / b[0])

#*********  Cusp Anomalous Dims. *************
    

# The coefficients j_F and \kappa_F
# ---------------------------------
    
jH = 1
def jJ_a(loc): return 2 - avec[loc]
jS = 1

kH = 4 
def kJ_a(loc): return - 2 / (1 - avec[loc])
def kS_a(loc): return + 4 / (1 - avec[loc])

# The cusp anomalous dimension 
# ----------------------------
    
Gq = np.zeros(4)

Gq[0] = 4 * CF
Gq[1] = 4 * CF * ( (67 / 9 - (pi**2)/ 3 ) * CA - (20 / 9) * TF * nf  )
Gq[2] = 4 * CF * ( (245 / 6 - 134 / 27 * (pi**2) + 11 / 45 * (pi**4)          \
      + 22 / 3 * zeta(3) ) * (CA ** 2)  + ( 40 / 27 * (pi**2)- 418 / 27       \
      - 56 / 3 * zeta(3)) * CA * TF * nf + (16 * zeta(3) - 55/3) *CF* TF * nf \
      - 16 / 27 * TF**2 * nf**2 ) 
Gq[3] = Gq[2]**2 / Gq[1]


# The individual cusp terms 
# -------------------------

GH = -  jH * kH / 2 * Gq                                      # Hard

def GJ_a(loc): return - Gq * jJ_a(loc) * kJ_a(loc)  / 2        # Jet

def GS_a(loc): return - Gq * jS * kS_a(loc)  / 2           # Soft 

# *************** Total cross section at each order *****************
sigma_tot = np.zeros(4)

sigma_tot[0] = 1

sigma_tot[1] = (3 / 2) * CF

sigma_tot[2] = - 3 / 8 * (CF**2) + (123 / 8 - 11 * zeta(3)) * (CF*CA)         \
             + (-11 / 2 + 4 * zeta(3)) * (CF*TF*nf)
            
sigma_tot[3] = - 0.411757 * (2 * pi)**3

#***********  Fixed order constants *************

# Hard Function
# -------------

cH = np.zeros(3)

cH[1] = (7 / 3 * pi**2 - 16) * CF
cH[2] = (511 / 4 - 83 / 3 * pi**2 + 67 / 30 * pi**4 - 60 * zeta(3)) * CF**2   \
      +(- 51157/324+ 1061/54*pi**2 - 8/45 * pi**4 + 626/9 * zeta(3)) * CF*CA  \
      + (4085 / 81 - 182 / 27 * pi**2 + 8 / 9 * zeta(3)) * CF * TF * nf


# Soft Function (Laplace space)
# -----------------------------
     
c2CA_full = np.array([-22.430,-29.170,-36.398,-44.962,-56.499,-74.717,-110.55]) 
    
c2nf_full =  np.array([27.315, 28.896, 31.589, 36.016, 43.391, 56.501, 83.670])

def cS_a(loc):

  c2CA  = c2CA_full[loc] 
  c2nf  = c2nf_full[loc]

  cS_0 = 0
  cS_1 = -(CF * pi**2) * (1 / (1-avec[loc]))
  cS_2 = c2CA * CA *CF + c2nf * CF*TF*nf + CF**2 * pi**4 / 2  / (1-avec[loc])**2

  return np.array( [cS_0, cS_1, cS_2] )


# Jet Function (Laplace space)
# ----------------------------

fJ_full   = np.array([-1.8236, -1.4416, -1.02788, -0.561666, 0., 0.760554, 2.0393])
cJ_2_full = np.array([62.2698, 39.7567, 15.75, -10.0915, -36.4332 , -54.7994, -25.2242])


def cJ_a(loc):

  fJ = fJ_full[loc]

  cJ_0 = 0.0
  cJ_1 = CF / (2 - avec[loc]) * (14 - 13 * avec[loc] -pi**2 / 6 * (8 - 20 * avec[loc]         \
      + 9 * avec[loc]**2) / (1 - avec[loc]) - 4 * fJ )
  cJ_2 = cJ_2_full[loc]

  return np.array( [cJ_0, cJ_1, cJ_2] )   

#*******  Non-Cusp Anomalous Dims. ***********

# Hard Function
# -------------

gH = np.zeros(3)

gH[0] = - 12 * CF
gH[1] = - 2  * CF * ( (82 / 9 - 52 * zeta(3)) * CA + (3 - 4 * pi**2           \
      + 48 * zeta(3)) * CF + (65 / 9 + pi**2) * b[0]) 
gH[2] = - 4  * CF * ((66167 / 324 - 686 / 81 * pi**2 - 302 / 135 * pi**4      \
      - 782 / 9 * zeta(3) + 44 / 9 * pi**2 * zeta(3) + 136 * zeta(5)) * CA**2 \
      + (151 / 4 - 205 / 9 * pi**2 - 247 / 135 * pi**4 + 844 / 3 * zeta(3)    \
      + 8 / 3 * pi**2 * zeta(3) + 120 * zeta(5)) * CF * CA                    \
      + (29 / 2 + 3 * pi**2 + 8 / 5 * pi**4 + 68 * zeta(3)                    \
      - 16 / 3 * pi**2 * zeta(3)- 240 * zeta(5))                              \
      * CF**2 + (-10781 / 108 + 446 / 81 * pi**2 + 449 / 270 * pi**4          \
      - 1166 / 9 * zeta(3)) * CA * b[0] + (2953 / 108 - 13 / 18 * pi**2       \
      - 7 / 27 * pi**4 + 128 / 9 * zeta(3) ) * b[1] + (- 2417 / 324           \
      + 5 / 6 * pi**2 + 2 / 3 * zeta(3)) * b[0]**2 )
      

# Soft Function (a=0)
# -------------------

gSa0 = np.zeros(3)

gSa0[0] = 0
gSa0[1] = -2 * CF*((64 / 9 - 28 * zeta(3)) * CA + (56 / 9 - pi**2 / 3) * b[0])
gSa0[2] = -2 * CF*((37871 / 162 - 310 / 81 * pi**2 - 8 / 5 * pi**4 - 2548 / 9 \
        * zeta(3) + 88 / 9 * pi**2 * zeta(3) + 192 * zeta(5) ) * CA**2        \
        + (-4697 / 54 - 242 / 81 * pi**2 + 56 / 45 * pi**4                    \
        - 220 / 9 * zeta(3)) * CA * b[0] + (1711 / 54 - pi**2 / 3 - 4 / 45    \
        * pi**4 - 152 / 9 * zeta(3) ) * b[1] + (-520 / 81 - 5 /9 * pi**2      \
        + 28 / 3 * zeta(3)) * b[0]**2 )
      
      
# Soft Function (a!=0)
# --------------------

g1CA_full = np.array([1.04174, 5.86486, 9.89755, 13.1901, 15.7945, 17.7606, 19.1316])

g1nf_full = np.array([-0.957103, 0.528432, 1.84398, 2.97513, 3.90981, 4.63981, 5.16128])
    

def gS_a(loc):

  g1CA = g1CA_full[loc]
  g1nf = g1nf_full[loc]

  gS_0 = 0
  gS_1 = (2 / (1 - avec[loc])) * (g1CA * CF * CA + g1nf * CF * TF * nf )
  gS_2 = 0       #  Unknown value 

  return np.array([gS_0, gS_1, gS_2])
 


# Jet Function (a!=0)
# -------------------

def gJ_a(loc): return - ( gH + gS_a(loc) ) / 2 

# Gap parameter
# -------------

def gR_a(loc):

  gR_0 = 0
  gR_1 = exp(euler_gamma) / 2 * (gS_a(loc)[1] + 2 * cS_a(loc)[1] * b[0])
  gR_2 = 0 #  Unknown value 

  return np.array( [gR_0, gR_1, gR_2] )
