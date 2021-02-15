###############################################################################
from alphaS import *
from special_functions import *
from remainder_functions import *
from param import *
from renormalization_group import *
from profiles import *
import kernels
from newton_optimize import *
import os
import sys
import glob
from numpy import pi, log, exp, real, absolute, euler_gamma
from scipy.special import psi, binom, gamma
from scipy.integrate import newton_cotes as  nc
from scipy.linalg import block_diag
from scipy import interpolate
from iminuit import Minuit

###############################################################################

# Model function f_mod
# --------------------

def fmod(x, l):
    return(128/3) * x**3 / l**3 / absolute(l)  *  exp(-4*x/l) *  np.heaviside(x,0) # 

def Dfmod(x, l):
    return  (128/3) * x**2 / l**4 / absolute(l)   *  exp(-4*x/l) * (3 * l - 4 *\
            x) * np.heaviside(x,0)  # 
    
def DDfmod(x, l):
    return (256/3) * x / l**5 / absolute(l)   * exp(-4*x/l) * \
 (3 * l**2 - 12 * l * x + 8 * x**2) *  np.heaviside(x,0)  # 

# Subtraction term 
# ----------------

# The functions delta1 and delta2 return an array dimension size_a. The entries 
# correspond to the values of a. 
    
def delta1(m, R):
    return GS[0] * log(m / R)


def delta2(m, R):
    return GS[0] * b[0] * (log(m / R))**2 + GS[1] * log(m / R) + gS[1] / 2    \
    + cS[1] * b[0]

# The functions f^{(i)}_{mod}
# ---------------------------
    
def f0(k,m,R,RStar,l):
    return fmod(k,l)

def f1(k,m,R,RStar,l):
    return - a_s(m) / (4 * pi) * 2 * delta1(m,R) * RStar * exp(euler_gamma)   \
    * Dfmod(k,l)

def f2(k,m,R,RStar,l):
    return  (a_s(m) / (4 * pi))**2 * ( -2 * delta2(m,R) * RStar              \
    * exp(euler_gamma) * Dfmod(k,l) + 2  * (delta1(m,R) * RStar              \
    * exp(euler_gamma))**2 * DDfmod(k,l) )

# Moments of the functions f^{(i)}_{mod}
# --------------------------------------
    
def f0M(m,R,RStar,l):
    return l
             
def f1M(m,R,RStar,l):
    return + a_s(m) / (4 * pi) * 2 * delta1(m,R) * RStar * exp(euler_gamma)
    
def f2M(m,R,RStar,l):
    return  (a_s(m) / (4 * pi))**2 * 2 * delta2(m,R) * RStar * exp(euler_gamma)  

###############################################################################  
# Treatment of fixe order terms 
# -----------------------------  

def Lp(k,L,O,cT):
    binomial_epxand = []
    for n in range(k+1):
        c = cT * binom(k,n) * (L**n)
        binomial_epxand.append( partial(O, k-n, c))
    return(binomial_epxand)        
#-----------------------------------
def add(list_of_partial_derivatives):
    sum = 0
    for p in list_of_partial_derivatives:
        sum += p.d_w()
    return sum
#-----------------------------------
def list_outer(list_1, list_2):
    n1 = len(list_1)
    n2 = len(list_2)
    outer = []
    for p1 in list_1:
        for p2 in list_2:
            outer.append(p1 * p2)
    return outer

# Measured fixed order terms including \partial_{\Omega}
# ------------------------------------------------------
    
def F_meas(L,O,G,g,j,c):
    F1 =  Lp(2,L,O, G[0]/ j**2)  + Lp(1,L,O,g[0] /j) + Lp(0,L,O, c[1])
    F2 =  Lp(4,L,O, G[0]**2 / 2 / j**4 )                                      \
    + Lp(3,L,O, G[0] * (g[0] + 2 * b[0] / 3 ) / j**3 )                        \
    + Lp(2,L,O,(G[1] + g[0]**2 / 2 + g[0] * b[0] + c[1] * G[0]) / j**2)       \
    + Lp(1,L,O, (g[1] + c[1] * g[0] + 2 * c[1] * b[0] ) / j )                 \
    + Lp(0,L,O, c[2])
    return [ 1, F1, F2]

# Unmeasured fixed order terms
# ----------------------------
    
def F_unme(L,G,g,j,c):
    F1 = G[0] * (L / j)**2 + g[0] * (L / j) + c[1]
    F2 = (G[0]**2 / 2) * (L / j)**4                                           \
    + G[0] * (g[0] + 2 * b[0] / 3) * (L / j)**3                               \
    + (G[1] + g[0]**2 / 2 + g[0] * b[0] + c[1] * G[0]) * (L / j)**2           \
    + (g[1] + c[1] * g[0] + 2 * c[1] * b[0] ) * (L / j)                       \
    + c[2] 
    return [ 1, F1, F2]

# Measured and unmeasured logs 
# ----------------------------

def L_unme(mH,Q):
    return log(mH / Q)

def L_meas(m,Q,j,tau):
    return log((m / Q)**j / tau)

# The function \tilde{F}_k from Eqs.(4.39) 
# ----------------------------------------
    
def FT(m,mH,mJ,mS,Q,tau):

    O = Omega(m,mJ,mS)
    LH = L_unme(mH,Q)
    LJ = L_meas(mJ,Q,jJ,tau)
    LS = L_meas(mS,Q,jS,tau)
    H = F_unme(LH,GH,gH,jH,cH)
    J1 = add( F_meas(LJ,O,GJ,gJ,jJ,cJ)[1] )
    S1 = add( F_meas(LS,O,GS,gS,jS,cS)[1] )
    J2 = add( F_meas(LJ,O,GJ,gJ,jJ,cJ)[2] )
    S2 = add( F_meas(LS,O,GS,gS,jS,cS)[2] )
    J1_sqr = add(list_outer(F_meas(LJ,O,GJ,gJ,jJ,cJ)[1]                       \
                            ,F_meas(LJ,O,GJ,gJ,jJ,cJ)[1]))
    J1_S1  = add(list_outer(F_meas(LJ,O,GJ,gJ,jJ,cJ)[1]                       \
                            ,F_meas(LS,O,GS,gS,jS,cS)[1]))
    aH = a_s(mH) / (4 * pi) 
    aJ = a_s(mJ) / (4 * pi) 
    aS = a_s(mS) / (4 * pi) 
    F1 = aH * H[1] +  2 * aJ * J1 + 1 *  aS * S1
    F2 = aH**2 * H[2] + aJ**2 * (2 * J2 + J1_sqr) + aS**2 * S2                \
    +  aH * H[1] *(2 * aJ * J1 + aS * S1) + 2 * aS * aJ * J1_S1
    return [ 1, F1, F2]

###############################################################################
# Perturbative cross sections
# ---------------------------

# \sigma_c^{(i)} without matching
# -------------------------------
def sigma_c_0(m,mH,mJ,mS,Q,tau):
    f = FT(m,mH,mJ,mS,Q,tau)
    return exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                              \
    + euler_gamma * Omega(m,mJ,mS)) * f[0] * (1 / tau)** Omega(m,mJ,mS)       \
    /  gamma(1 - Omega(m,mJ,mS)) 
    
def sigma_c_1(m,mH,mJ,mS,Q,tau):
    f = FT(m,mH,mJ,mS,Q,tau)
    return  exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                             \
    + euler_gamma * Omega(m,mJ,mS)) * f[1] * (1 / tau)** Omega(m,mJ,mS)       \
    /  gamma(1 - Omega(m,mJ,mS))
    
def sigma_c_2(m,mH,mJ,mS,Q,tau):
    f = FT(m,mH,mJ,mS,Q,tau)
    return exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                              \
    + euler_gamma * Omega(m,mJ,mS)) * f[2] * (1 / tau)** Omega(m,mJ,mS)       \
    /  gamma(1 - Omega(m,mJ,mS))
    
def sigma_c(m,mH,mJ,mS,Q,tau):
    f = FT(m,mH,mJ,mS,Q,tau)
    return exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                              \
    + euler_gamma * Omega(m,mJ,mS)) * f * (1 / tau)** Omega(m,mJ,mS)          \
    /  gamma(1 - Omega(m,mJ,mS)) 

# The complete remainder functions
# --------------------------------
def r1_c(tau, m, loc ):
    return  a_s(m) * CF / (2 * pi) * r1(tau)[loc]

def r2_c(tau, m, Q, dr, loc):
    return (a_s(m) / (2 * pi))**2 * (r2(tau)[loc] \
               + b[0] * r1(tau)[loc] * log(m / Q) + dr * Dr2(tau)[loc] )

# Perturbative cross sections \sigma_c^{(i)} including  matching
# --------------------------------------------------------------
def sigma_c_1_NS(m,mH,mJ,mS,Q,mNS,tau,loc):
    f = FT(m,mH,mJ,mS,Q,tau)
    return  exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                             \
    + euler_gamma * Omega(m,mJ,mS)) * f[1] * (1 / tau)** Omega(m,mJ,mS)       \
    /  gamma(1 - Omega(m,mJ,mS)) + r1_c(tau, mNS, loc )
    
def sigma_c_2_NS(m,mH,mJ,mS,Q,mNS,tau,dr,loc):
    f = FT(m,mH,mJ,mS,Q,tau)
    return exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                              \
    + euler_gamma * Omega(m,mJ,mS)) * f[2] * (1 / tau)** Omega(m,mJ,mS)       \
    /  gamma(1 - Omega(m,mJ,mS)) + r2_c(tau, mNS, Q , dr, loc)
    
def sigma_c_NS(m,mH,mJ,mS,Q,mNS,tau,dr, loc):
    f = np.array(  FT(m,mH,mJ,mS,Q,tau) )
    return exp(TK(m,mH,mJ,mS,Q) + Tk(m,mH,mJ,mS)                              \
    + euler_gamma * Omega(m,mJ,mS)) * f * (1 / tau)** Omega(m,mJ,mS)          \
    /  gamma(1 - Omega(m,mJ,mS)) +                                            \
    np.array([0,r1_c(tau, mNS, loc), r2_c(tau, mNS, Q , dr, loc)]) 


# Total cross section
# -------------------
def sigmaT(Q, log_accuracy = 2):
    a = a_s(Q) / (2 * pi)

    s_tot = 0
    for order in range(log_accuracy + 1):
        s_tot += sigma_tot[order] * a**order

    return s_tot

###############################################################################
    
# Newton-Cotes Quadrature (integration algorithm) 
# -----------------------------------------------
def nc_integration( h, nc_points, f_points):
    F = np.array(f_points)
    if (nc_points == np.size(F)):
        w = h * nc(nc_points-1, equal= 1)[0]
        return np.sum(w * F)
    else : print("ERROR: invalid inputs for Newton - Cotes Quadrature" )
 
# returns the conceived cross section at the value \tau
# -----------------------------------------------------
def cross_section_shape(tau, loc, Q, Omega1, rnd_seed, log_accuracy = 2, i_max = 100, width_factor = 5, nc_points = 5):  
    
    l = (2 / (1 - avec[loc]) * (Omega1 - initial_D)) # \lambda (1st moment of f_{mod})
    
    # initializing profile functions
    # ------------------------------
    m = profile( Q, rnd_seed )
    
    # Profiles 
    # --------         
    mS = m.soft(tau, loc)
    mJ = m.jet(tau, loc)
    RS = m.R_scale(tau, loc)
    RStar =m.R_star(tau, loc)
    mNS = m.mNS(tau, loc)
    mH = m.hard()
    mF  = Q
    dr = 0   
 
    D = Da(mS , RS, RStar, RD, initial_D, loc)

    tau_min = 2 * D / Q

    e = 0.00000000001  # To avoit the exact end point 

    tau_max = min(tau,(2 * D + width_factor * 2 * (Omega1-initial_D)/(1-avec[loc]))/Q)-e

    h = (tau_max -tau_min) / i_max 
    hN = h / (nc_points -1)
    
    Jacobian = Q  
    
    # Loop over tau values tau_1 = tau_min , tau_{i_max} = tau_max
    # ------------------------------------------------------------
    f_0_0, f_0_1, f_0_2, f_1_0, f_1_1, f_2_0  = 0., 0., 0., 0., 0., 0.
    
    if (tau_min < tau_max) :
        
        for tauN in  np.linspace(tau_min, tau_max, num = i_max, endpoint = False):

            F_model = []
            sigmaC  = []
            
            for step in range(nc_points):

                F_model_k = [f0((tauN + step*hN) * Q - 2 * D, mS , RS, RStar, l), \
                             f1((tauN + step*hN) * Q - 2 * D, mS , RS, RStar, l), \
                             f2((tauN + step*hN) * Q - 2 * D, mS , RS, RStar, l) ]
                
                F_model.append(F_model_k)

                if (matched):
                    sigmaC_k = sigma_c_NS(mF,mH,mJ,mS,Q,mNS, tau-tauN-step*hN,dr, loc)
                else:
                    sigmaC_k = sigma_c(mF,mH,mJ,mS,Q,tau-tauN-step*hN)
                    
                sigmaC.append(sigmaC_k)

            sigmaC = np.transpose(sigmaC)
            F_model = np.transpose(F_model)
            #---------------------------     
            c_int, f_int = 0, 0
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_0_0 += nc_integration( hN, nc_points, f_points)
            #---------------------------
            c_int, f_int = 0, 1
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_0_1 += nc_integration( hN, nc_points, f_points)
            #---------------------------
            c_int, f_int = 0, 2
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_0_2 += nc_integration( hN, nc_points, f_points)
            #---------------------------
            c_int, f_int = 1, 0
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_1_0 += nc_integration( hN, nc_points, f_points)
            #---------------------------
            c_int, f_int = 1, 1
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_1_1 += nc_integration( hN, nc_points, f_points)
            #---------------------------
            c_int, f_int = 2, 0
            f_points = (sigmaC[c_int])*(F_model[f_int])
            f_2_0 += nc_integration( hN, nc_points, f_points)
                
        # ----- End of tau-loop -------
    
            
    if (log_accuracy == 2):          
        sigmaNP =  f_0_0 + (f_0_1 + f_1_0) + (f_0_2 + f_1_1 + f_2_0)
        # ------------------------------
    elif (log_accuracy == 1):        
        sigmaNP = f_0_0 + (f_0_1 + f_1_0)
        # ------------------------------    
    elif (log_accuracy == 0):        
        sigmaNP = f_0_0
        # ------------------------------
        
    return Jacobian * sigmaNP


def bin_it(bin_list, loc,  Q, Omega1, rnd_seed ):

    m = np.size(bin_list[:,0])

    x_points = np.append( bin_list[:,0] ,  bin_list[m-1,1] )

    bin_widths = bin_list[:,1] - bin_list[:,0]

    y_points = []

    for tau in x_points:
        y_points.append( cross_section_shape(tau, loc, Q, Omega1, rnd_seed) )
    binned_cross_section = []

    for i in range(m):
        binned_cross_section.append(  (y_points[i+1] - y_points[i])/bin_widths[i]  )

    return np.array(binned_cross_section)

###############################################################################

def set_chi_single_a( loc, Q, bins, Y_exp, V_inv, rnd_seed ):

    def chi(a_Z, Omega1):

        global a_s
        a_s = set_alpha_S(a_Z , 2)

        # Seting global variables 
        # -----------------------
        global kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR

        kJ = kJ_a(loc)
        jJ = jJ_a(loc)
        cJ = cJ_a(loc)
        GJ = GJ_a(loc)
        gJ = gJ_a(loc)

        kS = kS_a(loc)
        cS = cS_a(loc)
        GS = GS_a(loc)
        gS = gS_a(loc)

        gR = gR_a(loc)

        global Omega, TK, Tk, Da
        Omega, TK, Tk, Da = kernels.set_kernels(a_s, 2, kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR)

        Y_the = bin_it( bins, loc, Q, Omega1, rnd_seed) /  sigmaT(Q)

        D =Y_exp - Y_the

        return  np.dot(D, np.matmul (V_inv , D) )

    return chi

def set_chi_all_a( Q, bins, Y_exp, V_inv, rnd_seed, norms_exp = 1, normalization = False ):

    def chi( a_Z, Omega1):

        Y_the = np.array([])

        global a_s
        a_s = set_alpha_S(a_Z , 2)

        for loc in range(len(bins)):

            # Seting global variables 
            # -----------------------
            global kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR

            kJ = kJ_a(loc)
            jJ = jJ_a(loc)
            cJ = cJ_a(loc)
            GJ = GJ_a(loc)
            gJ = gJ_a(loc)

            kS = kS_a(loc)
            cS = cS_a(loc)
            GS = GS_a(loc)
            gS = gS_a(loc)

            gR = gR_a(loc)

            global Omega, TK, Tk, Da
            Omega, TK, Tk, Da = kernels.set_kernels(a_s, 2, kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR)

            bin_list = bins[loc]

            Y_the_temp = bin_it( bin_list, loc, Q, Omega1, rnd_seed) /  sigmaT(Q)

            if (normalization): norm = norms_exp[loc]/np.sum(Y_the_temp)
            else: norm = 1

            Y_the = np.concatenate( (Y_the, norm * Y_the_temp), axis = None)

        D = Y_exp - Y_the

        return  np.dot(D, np.matmul (V_inv , D) )

    return chi

def set_chi_thrust( bins, Y_exp, V_inv, Q_list, rnd_seed, norms_exp = 1, normalization = False ):

    def chi( a_Z, Omega1):

        Y_the = np.array([])

        global a_s
        a_s = set_alpha_S(a_Z , 2)

        loc = 4    # loc = 4 corresponds to thrust at perturbative level 

        # Seting global variables 
        # -----------------------
        global kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR

        kJ = kJ_a(loc)
        jJ = jJ_a(loc)
        cJ = cJ_a(loc)
        GJ = GJ_a(loc)
        gJ = gJ_a(loc)

        kS = kS_a(loc)
        cS = cS_a(loc)
        GS = GS_a(loc)
        gS = gS_a(loc)

        gR = gR_a(loc)

        global Omega, TK, Tk, Da
        Omega, TK, Tk, Da = kernels.set_kernels(a_s, 2, kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR)

        for k in range(len(Q_list)):

            bin_list = bins[k]

            Q = Q_list[k]

            Y_the_temp = bin_it( bin_list, loc, Q, Omega1, rnd_seed) /  sigmaT(Q)

            if (normalization): norm = norms_exp[k]/np.sum(Y_the_temp)
            else: norm = 1

            Y_the = np.concatenate( (Y_the, norm * Y_the_temp), axis = None)

        D = Y_exp - Y_the

        return  np.dot(D, np.matmul (V_inv , D) )

    return chi

###############################################################################

def DSigmaDTau(tau, loc, Q, rnd_seed, a_Z =0.11, Omega1 = 0.4):
    # Variables that normaly would be set by a run_cart.sh 
    # Un-comment only for testing purpuses
    # ----------------------------------------------------

    # set the strong coupling 
    # -----------------------
    global a_s
    a_s = set_alpha_S(a_Z , log_accuracy)

    for loc in [4]:

        # Seting global variables 
        # -----------------------
        global kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR

        kJ = kJ_a(loc)
        jJ = jJ_a(loc)
        cJ = cJ_a(loc)
        GJ = GJ_a(loc)
        gJ = gJ_a(loc)

        kS = kS_a(loc)
        cS = cS_a(loc)
        GS = GS_a(loc)
        gS = gS_a(loc)

        gR = gR_a(loc)

        global Omega, TK, Tk, Da
        Omega, TK, Tk, Da = kernels.set_kernels(a_s, 2, kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR)

        return cross_section_shape(tau, loc, Q, Omega1, rnd_seed) 

###############################################################################

#--------------------------------------- 
def experimental_input_all_a( bin_min, num_of_bins, include_norms = False) :

    bin_max = bin_min + num_of_bins 

    Y_exp = np.array([])
    S_exp = np.array([])
    E_exp = np.array([])
    bins  = []
    V_temp = []
    norms_exp = []

    for loc in range(7): 

        # Experimental data
        # -----------------
        dataFile = "../exp_data/exp_data_" + str(loc) + ".txt"
        exp_data = np.genfromtxt(dataFile , delimiter='\t') 
        Y_exp = np.concatenate((Y_exp,  exp_data[bin_min[loc]: bin_max[loc],2]),     axis = 0)
        norms_exp.append( np.sum(exp_data[bin_min[loc]: bin_max[loc],2]) )
        # s     = np.concatenate((s,      np.arange(bin_min[loc], bin_max[loc])  ), axis = 0)
        E_exp = (exp_data[bin_min[loc]: bin_max[loc], 4])**2
        S_exp = np.concatenate((S_exp, (exp_data[bin_min[loc]: bin_max[loc],3])**2), axis = 0)
        bins.append( exp_data[bin_min[loc]: bin_max[loc],[0,1]] )

        # Covariance matrix (minimal-overlasp model)
    # -----------------------------------------
        num_of_points = num_of_bins[loc]
        
        A = np.zeros((num_of_points, num_of_points))
        for i in range(num_of_points):
            for j in range (num_of_points):
                if (i == j) :
                    A[i,i] = E_exp[i]
                else:
                    A[i,j] = min(E_exp[i],E_exp[j])

        V_temp.append(A)

    # cov_matrix = "../inputs/cov.txt"
    # S = np.genfromtxt(cov_matrix , delimiter='\t') 
    # s = s.astype(int)
    # S_exp = select(S, s, s)

    S_exp = np.diag(S_exp)

    V = block_diag(*V_temp) + S_exp

    V_inv = np.linalg.inv(V)

    if (include_norms): return [bins, Y_exp, V_inv, norms_exp]
    else:  return [bins, Y_exp, V_inv]

#--------------------------------------- 

def experimental_input_single_a( loc, bin_min, num_of_bins ):

    # Seting global variables 
    # -----------------------
    global kJ, jJ, cJ, GJ, gJ, kS, cS, GS, gS, gR

    kJ = kJ_a(loc)
    jJ = jJ_a(loc)
    cJ = cJ_a(loc)
    GJ = GJ_a(loc)
    gJ = gJ_a(loc)

    kS = kS_a(loc)
    cS = cS_a(loc)
    GS = GS_a(loc)
    gS = gS_a(loc)

    gR = gR_a(loc)

    # Experimental data
    # -----------------
    bin_max = bin_min + num_of_bins 
    dataFile = "../exp_data/exp_data_" + str(loc) + ".txt"
    exp_data = np.genfromtxt(dataFile , delimiter='\t') 
    Y_exp = exp_data[bin_min: bin_max, 2]
    bins  = exp_data[bin_min: bin_max,[0,1]]

    # Covariance matrix (minimal-overlasp model)
    # -----------------------------------------
    statistical =  (exp_data[bin_min: bin_max,4])**2
    systematic  =  (exp_data[bin_min: bin_max,6])**2
    V = np.zeros((num_of_bins,num_of_bins))
    for i in range(num_of_bins):
        for j in range (num_of_bins):
            if (i == j) :
                V[i,i] = statistical[i] + systematic[i]
            else:
                V[i,j] = min( systematic[i], systematic[j] )
    V_inv = np.linalg.inv(V)

    V_inv = np.linalg.inv(V)

    return [bins, Y_exp, V_inv]


#--------------------------------------- 
def experimental_input_thrust(include_norms = False) :

    Y_exp = np.array([])
    S_exp = np.array([])
    E_exp = np.array([])
    bins  = []
    V_temp = []
    norms_exp = []
    Q_list = []

    input_name = "../exp_data/thrust/*select.txt"

    list=glob.glob(input_name)

    for dataFile in list:
        Q_list.append( float(dataFile.split("_")[-2].replace("p",".")) )

        # Experimental data
        # -----------------
        exp_data = np.genfromtxt(dataFile , delimiter='\t') 
        Y_exp = np.concatenate((Y_exp,  exp_data[:,3]),     axis = 0)
        norms_exp.append( np.sum(exp_data[:,3]) )
        E_exp = (exp_data[:, 5])**2
        S_exp = np.concatenate((S_exp, (exp_data[:,4])**2), axis = 0)
        if (exp_data[0,1] > exp_data[0,2] ): bins_temp = exp_data[:,[2,1]]
        else: bins_temp = exp_data[:,[1,2]]
        bins.append(  bins_temp )

        # Covariance matrix (minimal-overlasp model)
        # -----------------------------------------
        num_of_points = len(bins_temp) 
        
        A = np.zeros((num_of_points, num_of_points))
        for i in range(num_of_points):
            for j in range (num_of_points):
                if (i == j) :
                    A[i,i] = E_exp[i]
                else:
                    A[i,j] = min(E_exp[i],E_exp[j])

        V_temp.append(A)

    S_exp = np.diag(S_exp)

    V = block_diag(*V_temp) + S_exp

    V_inv = np.linalg.inv(V)

    if (include_norms): return [bins, Y_exp, V_inv, Q_list, norms_exp]
    else:  return [bins, Y_exp, V_inv, Q_list]

#--------------------------------------- 