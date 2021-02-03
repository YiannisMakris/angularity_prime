import numpy as np 
from numpy import log, floor, euler_gamma, pi, exp, real, absolute
from scipy.special import  psi
from scipy.integrate import newton_cotes as  nc
import cmath

gammainc_data = np.genfromtxt('../inputs/Gammainc.txt', delimiter='\t') #import data 


def to_X(i):
    return 10. ** ((i+1)/1000 - 1)

def to_I(x):
    return 1000. * log(10 * x)/log(10) - 1


def Gammainc(x):
    
    if (isinstance(x, float) or isinstance(x, int) ):
        
        i = int(floor(to_I(x)))
        
        if (i < 0) or (i > 1999):
            print('ERROR: Gammainc evaluated outside region of \
                  definition')
        else:
            m1 = gammainc_data[i]
            m2 = gammainc_data[i+1]
            x1 = to_X(i)
            x2 = to_X(i+1)
            Dx = (x2 - x1)/(x - x1)
            Dm = m2 - m1
            return  (m1 + Dm / Dx)[0] + 1j * (m1 + Dm / Dx)[1]
    else:
        x = np.array(x)
        ginc = []
        i_list = floor(to_I(x)).astype(int)
        
        n = 0
        for i in i_list:
            if (i < 0) or (i > 1999):
                print('ERROR: Gammainc evaluated outside region of \
                      definition')
            else:
                m1 = gammainc_data[i]
                m2 = gammainc_data[i+1]
                x1 = to_X(i)
                x2 = to_X(i+1)
                Dx = (x2 - x1)/(x[n] - x1)
                Dm = m2 - m1
                ginc.append( (m1 + Dm / Dx)[0] + 1j * (m1 + Dm / Dx)[1] )
            n += 1
        return  ginc
    
    
    
def G_int(t_list,r):
    rtrn = np.zeros(np.size(t_list)) +0*1j
    i = 0
    for t in t_list:
        rtrn[i] = cmath.exp((r-2) * cmath.log(t) -t)
        i +=1
    return rtrn
    
def D_Gammainc(r1,r2,r3, nc_points):
    F = G_int(np.linspace(r2, r3, num = nc_points),r1)
    h = (r3 - r2)/(nc_points -1)
    w = h * nc(nc_points-1)[0]
    return np.sum(w * F)
    

    
def harmonic(x):
    return psi(1+x) + euler_gamma
    