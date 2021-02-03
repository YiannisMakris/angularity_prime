import numpy as np
from math import floor
from iminuit import Minuit
from scipy import interpolate
import time
import sys
from math import floor

#--------------------------------------- 
#  In house minimizer

def Derivative(f, f_central, x, y):

    h = 0.000001
    # Center point 
    fC      = f_central
    # Front-Back/Left-Right steps
    fPx     = f(x+h, y)
    fPy     = f(x, y+h)
    fMx     = f(x-h, y)
    fMy     = f(x, y-h)
    # Diagonal steps 
    fPxy    = f(x+h, y+h)
    fMxy    = f(x-h, y-h)
    # d1 = $\grad$   hess = Hessian Matrix
    d1   =  np.array([  (fPx - fMx)/(2*h), (fPy - fMy)/(2*h) ]) 
    hess =  np.array([[ (fPx + fMx -2*fC)/(h**2) ,\
                        (fPxy-fPx-fPy+2*fC-fMx-fMy+fMxy)/(2*h**2)  ],\
                      [ (fPxy-fPx-fPy+2*fC-fMx-fMy+fMxy)/(2*h**2)   ,\
                        (fPy + fMy -2*fC)/(h**2) ] ] )

    return [fC, d1, hess]


def newton_adaptive( f, x0 = 0.11, y0=0.4,  delta = 10 ):
    evaluations = 0
    point = [x0, y0]
    # print(point)
    fC = f( *point )
    evaluations +=1
    for i in range(400):
        fC_temp = fC
        der    = Derivative(f, fC_temp, *point)
        evaluations +=6
        step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
        point_temp = point + step
        fC = f( *point_temp )
        evaluations +=1
        counter = 0
        while (fC_temp - fC < 0): 
            delta /= 2
            step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
            point_temp = point + step
            fC = f( *point_temp )
            evaluations +=1
            counter += 1
            # print("Reduction of step. New delta = ", delta)
            # print(point_temp)
            if (counter > 5) : break
        point = point_temp
        # print(point_temp)
        if (np.absolute((fC-fC_temp)/fC) < 10**(-6)  ): 
            return [point, True, evaluations]
            break
    return[point, False]


def newton( f, x0 = 0.11, y0=0.4,  delta = 1 ):
    point = [x0, y0]
    print(point)
    fC = f( *point )
    evaluations = 1
    for i in range(400):
        fC_temp = fC
        der    = Derivative(f, fC, *point)
        evaluations += 6
        step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
        point  +=  step
        fC = f( *point)
        evaluations += 1
        print(point)
        counter = 0
        if (np.absolute((fC-fC_temp)/fC) < 10**(-6)  ): 
            return [point, True, evaluations]
            break
    return[point, False]


def bilateral_adaptive( f, x0 = 0.11, y0=0.4,  delta = 1 ):
    evaluations = 0
    point = [x0, y0]
    # print(point)
    fC = f( *point )
    evaluations +=1
    for i in range(400):
        enterIN = False
        enterDE = False
        der    = Derivative(f, fC, *point)
        evaluations +=6
        step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
        point_temp = point + step
        fC_pre = fC
        fC_temp = fC
        fC = f( *point_temp )
        evaluations +=1
        counter = 0

        while (fC_pre - fC < 0): 
            enterDE = True
            delta /= 2
            # print( "De-creasing step. delta =  " , delta)
            step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
            point_temp = point + step
            fC = f( *point_temp )
            evaluations +=1
            counter += 1
            if (counter > 5) : break

        if (not enterDE):
            while  (fC_temp - fC > 0):
                # Confirms it enters the loop 
                enterIN = True
                # Saving valus from last step
                delta_temp = delta
                point_temp = point + step
                fC_temp = fC
                # Calculating new step and updating values
                # print( "In-creasing step. delta =  " , delta)
                delta *= 2
                step   = - delta * np.matmul(np.linalg.inv(der[2]), der[1] )
                point_new = point + step
                fC = f( *point_new )
                evaluations +=1

        # Check if entered the "increasing" loop and adjust for fC
        if (enterIN): 
            fC = fC_temp
            delta = delta_temp

        point = point_temp

        # print(point_temp)
        if (np.absolute((fC-fC_pre)/fC) < 10**(-6)  ): 
            return [point, True, evaluations]
            break
    return[point, False]

def scan_interpolation(f, x_min, x_max, y_min, y_max, f_init = 10**6, gridSize = 5, delta = 1):

    x_list = np.linspace(x_min, x_max, num = gridSize, endpoint = True)
    y_list = np.linspace(y_min, y_max, num = gridSize, endpoint = True)

    f_min = f_init

    grid  = []

    counter = 0
    for x in x_list:
        for y in y_list:
            print("Scan progress: ", floor(counter/gridSize**2*1000)/10, " %", end = '\r')

            counter += 1

            z = f(x,y)
            time.sleep(0.1)



            grid.append([ x, y, z] )

            if (z < f_min):
                f_min = z
                point = [x, y]

            sys.stdout.write("\033[K")

    grid = np.array(grid)
    tck = interpolate.bisplrep(grid[:,0] , grid[:,1], grid[:,2])
    def inter(x, y): return interpolate.bisplev(x, y, tck)

    m = Minuit(inter, *point)
    m.migrad()  # run optimiser
    minimum= np.array(m.values) 

    # minimum = GD_adaptive(inter, *point)
    return minimum








# -------  Test Function -------

def f(x,y): return (20 * x-3)**2 + (y-1)**2+10


# print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
# print(" ++++++++++  Newton_adaptive ++++++++++ ")

# print(Newton_adaptive(f, 0,0))

# print(" +++++++++++++++++++++++++++++++++++++++")
# print(" ++++++++++    GD_adaptive    ++++++++++ ")

# print(GD_adaptive(f, 0,0))

# print(" +++++++++++++++++++++++++++++++++++++++")
# print(" ++++++++++       Newton      ++++++++++ ")

# print(Newton(f, 0,0))

# print(" +++++++++++++++++++++++++++++++++++++++")
# print(" ++++++++++       Minuit      ++++++++++ ")
# m = Minuit(f, 0,0)
# m.migrad()  # run optimiser
# minimum= np.array(m.values) 
# print(minimum)

# print(" +++++++++++++++++++++++++++++++++++++++")
# print(" +++++++++ scan_interpolation ++++++++++ ")

# print(scan_interpolation(f, -1, 1, -2, 2, gridSize = 11))

# print(" +++++++++++++++++++++++++++++++++++++++")




