import numpy as np 
from numpy import log, floor, pi

#------------------------------------------------------------------------------
r1_data  = np.genfromtxt('../inputs/r1.txt' , delimiter='\t') #import data for r1
r2_data  = np.genfromtxt('../inputs/r2.txt' , delimiter='\t') #import data for r2
Dr2_data = np.genfromtxt('../inputs/Dr2.txt', delimiter='\t') #import data for r2 error
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def to_x(i):
    xmin = 0.0
    xmax = .9812
    n = 10000
    Dx = xmax - xmin
    l = n / Dx
    beta = - l * xmin
    return (i - beta)/ l

def to_i(x):
    xmin = 0.0
    xmax = 0.9812
    n = 10000
    Dx = xmax - xmin
    l = n / Dx
    beta = - l * xmin
    return  (floor(l * x + beta)).astype(int) 
#------------------------------------------------------------------------------
def r1(x):
    i = to_i(x)
    if (i < 0) or (i > 10000):
        print('ERROR: NNLO remainder function evaluated outside region of \
                  definition')
    else:
        m1 = r1_data[i]
        m2 = r1_data[i+1]
        x1 = to_x(i)
        x2 = to_x(i+1)
        Dx = (x2 - x1)/(x - x1)
        Dm = m2 - m1
        return  np.array(m1 + Dm / Dx)
#------------------------------------------------------------------------------    
def r2(x):
    i = to_i(x)
    if (i < 0) or (i > 10000):
        print('ERROR: NNLO remainder function evaluated outside region of \
                  definition')
    else:
        m1 = r2_data[i]
        m2 = r2_data[i+1]
        x1 = to_x(i)
        x2 = to_x(i+1)
        Dx = (x2 - x1)/(x - x1)
        Dm = m2 - m1
        return  np.array(m1 + Dm / Dx)
#------------------------------------------------------------------------------
def Dr2(x):
    i = to_i(x)
    if (i < 0) or (i > 10000):
        print('ERROR: NNLO remainder function error evaluated outside region \
              of definition')
    else:
        m1 = Dr2_data[i]
        m2 = Dr2_data[i+1]
        x1 = to_x(i)
        x2 = to_x(i+1)
        Dx = (x2 - x1)/(x - x1)
        Dm = m2 - m1
        return  np.array(m1 + Dm / Dx)



    