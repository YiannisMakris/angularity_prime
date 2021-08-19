import numpy as np
from numpy import pi, log, exp, real
from param import m_Z, log_accuracy
from renormalization_group import b



#*********  Coupling up to four-loops  *************

def set_alpha_S(a_Z, log_accuracy = log_accuracy):
    
    def X(m): 
        return 1 + a_Z * b[0] / (2 * pi) * log(m / m_Z)
    
    
    if (log_accuracy == 0 ) :
        def a_s(m):
            x = X(m)
            a = a_Z / (4 * pi)
            Raa = x 
            return Raa**(-1) * a_Z
    elif (log_accuracy == 1 ):
        def a_s(m):
            x = X(m)
            a = a_Z / (4 * pi)
            Raa = x + a * b[1] / b[0] * log(x) 
            return Raa**(-1) * a_Z
    elif (log_accuracy == 2 ):
        def a_s(m):
            x = X(m)
            a = a_Z / (4 * pi)
            Raa = x + a * b[1] / b[0] * log(x)                                    \
            + (a**2) * ((b[2] / b[0]) * (1 - 1/x) + (b[1] / b[0])**2 * (log(x)/x  \
            + 1/x -1) )
            return Raa**(-1) * a_Z
    elif (log_accuracy == 3 ):
        def a_s(m):
            x = X(m)
            a = a_Z / (4 * pi)
            Raa = x + a * b[1] / b[0] * log(x)                                    \
            + (a**2) * ((b[2] / b[0]) * (1 - 1/x) + (b[1] / b[0])**2 * (log(x)/x  \
            + 1/x -1) )                                                           \
            + (a**3) / (x**2)  * ( b[3] / (2 * b[0]) * (x**2 - 1)                 \
            + b[1] * b[2] / (b[0]**2) * (x + log(x) - x**2) + b[1]**3             \
            / (2 * b[0]**3) * ((1-x)**2 - (log(x))**2 ))
            return Raa**(-1) * a_Z
    else:
        def a_s(m):
            x = X(m)
            a = a_Z / (4 * pi)
            Raa = x + a * b[1] / b[0] * log(x)                                    \
            + (a**2) * ((b[2] / b[0]) * (1 - 1/x) + (b[1] / b[0])**2 * (log(x)/x  \
            + 1/x -1) )                                                           \
            + (a**3) / (x**2)  * ( b[3] / (2 * b[0]) * (x**2 - 1)                 \
            + b[1] * b[2] / (b[0]**2) * (x + log(x) - x**2) + b[1]**3             \
            / (2 * b[0]**3) * ((1-x)**2 - (log(x))**2 ))
            return Raa**(-1) * a_Z
        print("Warning: from set a_s running: running coupling was set to four \
    loops running.  ")
        
    return a_s
    

