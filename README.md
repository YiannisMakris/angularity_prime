# angularity_prime

A python code for the extraction of the strong coupling constant based on chi-squared fit of angularities distributions. 

The experimental distribution from L3 collaboration measurements in 2009 is compared against the QCD NNLL-prime predictions. 

A two dimensional fit by minimizing chi-squared is obtained for the strong coupling constant and the hadronization parameter Omega_1.

The minimization of the chi-squared distribution is obtained by:

Constructing a 2-D grid (alpha_S, Omega_1) ---> Interpolation of the grid is perfomed using scipy.interpolate.bisplev ---> Minimize interpolated function using mygrad() method from  iinuit.Munuit 
