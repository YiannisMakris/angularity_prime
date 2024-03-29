import numpy as np 
import random as rnd
from param import avec, m_Z, Q
from scipy.special import polygamma, hyp2f1 
from special_functions import *

tsph_full = 2 / (4 - avec) * hyp2f1(1, - avec / 2, 3 - avec / 2, -1)  #symmetric tau 

class profile:
    
    def __init__(self, Q, rnd_seed, hard_central = False):

        self.rnd_seed = rnd_seed
        self.Q = Q
        self.useStar = False
        
        rnd.seed( self.rnd_seed ) # sets the random seed for the non-central profiles (used in case of parallelization) 

        if (  self.rnd_seed < 0) :
            self.eH = 1.
            self.eJ = 0.
            self.eS = 0.
            self.n0 = 1.5
            self.n1 = 10.
            self.n2 = 1.
            self.n3 = 0.85
            self.m0 = 1.
            self.R0 = self.m0 - 0.4
            self.r  = 1.
            self.dc = 0
            self.dr = 0
            self.ns = 0
        else :     
            if hard_central:
                self.eH = 1
            else:
                self.eH = rnd.uniform(0.5, 2)
            self.eJ = rnd.uniform(-0.71, 0.75)
            self.eS = 0
            self.n0 = rnd.uniform(1, 2)
            self.n1 = rnd.uniform(8.5, 11.5)
            self.n2 = rnd.uniform(0.9, 1.1)
            self.n3 = rnd.uniform(0.8, 0.9)
            self.m0 = rnd.uniform(0.8, 1.2)
            self.R0 = self.m0 - 0.4
            self.r  = rnd.uniform(0.75, 1.33)
            self.dc = rnd.uniform(-1., 1.)
            self.dr = rnd.uniform(-1., 1.)
            self.ns = rnd.choice((-1, 0, 1))
            
        #-----------------------------------
        self.t0 = self.n0 / Q * 3.**avec
        self.t1 = self.n1 / Q * 3.**avec
        self.t2 = self.n2 * 0.295**(1 - 0.637 * avec)
        self.t3 = self.n3 * tsph_full
        
    def hard(self):
        return self.Q * self.eH
    #-----------------------------------
    @staticmethod
    def zi(tau,t0,y0,r0,t1,y1,r1):
        t_mean = (t0 + t1) / 2.
        a = y0 + r0 * t0 
        A = y1 + r1 * t1
        c = 2 * (A - a) / ((t0 - t1)**2) + (3. * r0 + r1) / (2. * (t0 - t1))
        C = 2 * (a - A) / ((t0 - t1)**2) + (3. * r1 + r0) / (2. * (t1 - t0))
        if tau < t_mean :
            return a + r0 * (tau - t0) + c * (tau - t0)**2
        else:
            return A + r1 * (tau - t1) + C * (tau - t1)**2
        
    #-----------------------------------
    def mrun(self, tau, a):       
        if tau <= self.t0[a] :
            return self.m0
        elif (self.t0[a] < tau) and (tau <= self.t1[a]):
            return self.zi(tau,self.t0[a],self.m0,0,self.t1[a],0, self.r      \
            * self.hard()/tsph_full[a])
        elif (self.t1[a] < tau) and (tau <= self.t2[a]):
            return self.r * self.hard() * tau / tsph_full[a]
        elif (self.t2[a] < tau) and (tau <= self.t3[a]):
            return self.zi(tau, self.t2[a], 0, self.r * self.hard()/tsph_full[a],  \
            self.t3[a],  self.hard(),0)
        else :
            return self.hard()
    #-----------------------------------    
    def mrun_R(self, tau, a):        
        if tau <= self.t0[a] :
            return self.R0
        elif (self.t0[a] < tau) and (tau <= self.t1[a]):
            return self.zi(tau,self.t0[a],self.R0,0,self.t1[a],0, self.r      \
            * self.hard()/tsph_full[a])
        elif (self.t1[a] < tau) and (tau <= self.t2[a]):
            return self.r * self.hard() * tau / tsph_full[a]
        elif (self.t2[a] < tau) and (tau <= self.t3[a]):
            return self.zi(tau, self.t2[a], 0, self.r * self.hard()/tsph_full[a],  \
            self.t3[a],  self.hard(),0)
        else :
            return self.hard()
    #-----------------------------------
    def soft(self, tau, a):       
        if tau < self.t3[a] :
            return (1 + self.eS * (1 - tau/self.t3[a])**2) * self.mrun(tau, a)
        else :
            return self.mrun(tau, a)
    #-----------------------------------    
    def jet(self, tau, a):        
        if tau < self.t3[a] :
            return (1 + self.eJ * (1 - tau/self.t3[a])**2)                    \
            * (self.mrun(tau, a))**(1/(2. - avec[a])) * (self.hard())**((1    \
            - avec[a])/(2. - avec[a]))
        else :
            return self.mrun(tau, a)
    #-----------------------------------    
    def R_scale(self, tau, a):       
        if tau < self.t3[a] :
            return (1 + self.eS * (1 - tau/self.t3[a])**2) * self.mrun_R(tau,a)
        else :
            return self.mrun_R(tau, a)

    #-----------------------------------    
    def R_star(self, tau, a): 
        temp_R  = self.R_scale( tau, a)

        if (self.useStar): temp_RM = self.R_scale( self.t1[a] , a)
        else: temp_RM = temp_R

        if ( tau  <  self.t1[a]  ): return temp_R
        else: return temp_RM


    #-----------------------------------   
    def mNS(self, tau, a):        
        if self.ns == 1:
            return (self.hard() + self.jet(tau, a)) / 2
        elif self.ns == 0:
            return self.hard()
        else:
            return (3 * self.hard() - self.jet(tau, a)) / 2

    def create_MIT_profile(self):
        return profile.MIT(self)


    class MIT:

        def __init__(self, profile):

            self.rnd_seed = profile.rnd_seed
            self.useStar = profile.useStar
            self.Q = profile.Q

            #-----------------------------------

            if (  self.rnd_seed < 0) :
                self.eH = 1.
                self.eJ = 0.
                self.n1 = 10.
                self.t2 = 0.295
                self.m0 = 1.
                self.R0 = self.m0 * (0.85)
                self.ns = 0
            else :     
                self.eH = 2**(rnd.uniform(-1, 1))
                self.eJ = rnd.uniform(-1, 1)
                self.n1 = rnd.uniform(2, 8)
                self.t2 = rnd.uniform(0.2, 0.3)
                self.m0 = rnd.uniform(1.5, 2.5)
                self.R0 = self.m0 * (0.85)
                self.ns = rnd.choice((-1, 0, 1))

            self.t1 = self.n1 / self.Q
            self.b  = 2* ( self.Q * self.eH - self.m0) / (self.t2 -self.t1 + 0.5)
            self.d  = (self.m0 * (self.t2 + 0.5) - self.Q * self.eH * self.t1) / (self.t2 -self.t1 + 0.5)
            self.m1 = (2* self.d - 2* self.R0 + self.b * self.t1) / self.t1
            self.m2 = (-self.d + self.R0) / self.t1**2
 
        #-----------------------------------
        def hard(self):
            return self.Q * self.eH

        def soft(self,tau):
            if (tau <= self.t1):
                return self.m0 + self.b  * (tau**2)  / (2*self.t1) 
            elif (tau >= self.t2):
                return self.Q * self.eH - self.b * (0.5 - tau)**2 / (1-2*self.t2)
            else: 
                return self.b * tau +self.d

        def jet(self, tau):
            return (1+self.eJ * (0.5 - tau)**2) * ( self.hard() * self.soft(tau) )**(0.5)

        def R_scale(self, tau):
            if (tau <= self.t1):
                return self.R0 + self.m1 * tau +self.m2 * tau**2
            else : 
                return self.soft(tau)   
        
        def R_star(self, tau): 
            temp_R  = self.R_scale( tau)

            if (self.useStar): temp_RM = self.R_scale( self.t1 )
            else: temp_RM = temp_R

            if ( tau  <  self.t1  ): return temp_R
            else: return temp_RM

        def mNS(self, tau):        
            if self.ns == 1:
                return self.hard() 
            elif self.ns == 0:
                return self.jet(tau)
            else:
                return ( self.soft(tau) + self.jet(tau)) / 2

    #-----------------------------------
    #           END OF CLASS
    #-----------------------------------
###############################################################################
###############################################################################
#   Testing code goes here

# rnd_seed = -1

# m = profile( Q, rnd_seed )


# mit = m.create_MIT_profile()


# for i in range(6):
#     tau = 0.1 * i
#     mS = mit.soft(tau)
#     mJ = mit.jet(tau )
#     RS = mit.R_scale(tau)
#     mNS = mit.mNS(tau )
#     mH = mit.hard()
#     print(tau, "   ", RS, "   ", mS, "   ", mJ, "   ", mNS )


###############################################################################
###############################################################################
 
# Class for partil derivatives 
# ----------------------------
    
class partial:
    
    def __init__(self,  Omega, order = 1, coeff = 1):
        self.order = order 
        self.Omega = Omega 
        self.coeff = coeff
    
    #-----------------------------------
    
    def d_w(self):
        n = self.order
        O = self.Omega
        c = self.coeff
        
        if (n == 0):
            return c * 1
        elif (n == 1):
            return c * harmonic(-O)
        elif (n == 2):
            return c * ((harmonic(-O))**2 - polygamma(1,1 - O))
        elif (n == 3):
            return c * ((harmonic(-O))**3 - 3 * harmonic(-O)                  \
        * polygamma(1,1 - O) + polygamma(2, 1 - O))
        elif (n==4):
            return c * ( (harmonic(-O))**4 - 6 * ((harmonic(-O))**2)          \
        * polygamma(1,1 - O) + 4 * harmonic(-O) * polygamma(2,1 - O)          \
        + 3 * (polygamma(1,1 - O))**2  - polygamma(3, 1 - O) )
        else :
            return c * ( (harmonic(-O))**4 - 6 * ((harmonic(-O))**2)          \
        * polygamma(1,1 - O) + 4 * harmonic(-O) * polygamma(2,1 - O)          \
        + 3 * (polygamma(1,1 - O))**2  - polygamma(3, 1 - O) )
            
            print("WARNING: from partial: Beyond supported accuracy\
                  requested")
    
    #-----------------------------------
    
    def __mul__(self,other):
        if (self.Omega.all == other.Omega.all):
            return partial(self.Omega, self.order + other.order               \
                           , self.coeff * other.coeff)
        else:
            print("WARNING: from partial: incompatible conjunction of partial \
                  derivatives")
            return partial(self.Omega, self.order + other.order               \
                           , self.coeff * other.coeff)
            
    #-----------------------------------
        
    def __str__(self):
        n = self.order
        O = self.Omega
        c = self.coeff
        return "This is partial derivative of order n = {}, coefficient \
        c = {}, and Omega = {} ".format(n,c,O)
        
    #-----------------------------------
    #           END OF CLASS
    #-----------------------------------

