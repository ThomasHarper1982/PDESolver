
import numpy
from Network import *
from BasisFunction import *
from numpy import *
from numpy import matrix
import random
import copy


def flatten(o_l, limit=1000):
    new_l = copy.deepcopy(o_l)
    if limit > 0 and isinstance(o_l, list):
        new_l=[]
        for l in o_l:
            new_l+=flatten(l, limit=limit-1)
        return new_l
    else:
        return [new_l]

#random omega
def DerivationOmega(domain, target):
    omega1= map(lambda x: 30*random.random(), target)
    omega2= map(lambda x: 30*random.random(), target)

    omega1 = map(lambda x: x*(1-2*random.random()), omega1)
    omega2 = map(lambda x: x*(1-2*random.random()), omega2)
    return map(lambda o1,o2:[o1,o2] , omega1, omega2)
#grid
def DerivationPoints(domain):
    return copy.deepcopy(domain)

##The Interpolation Method function
##takes the domain and target values as 2 dimensional matrices
##sigma - the basis function, i.e. 'sigmoid'
##returns param - parameters (...,alpha_i, beta_i, omega1_i, omega2_i,...)
##for 0<i<n
def InterpolationMethod(domain_old, target_old, sigma):
    
    #we will treat Domain, Target as vectors
    domain = flatten(domain_old,limit=2)
    target = flatten(target_old)
    Omega  = DerivationOmega(domain, target)
    Points = DerivationPoints(domain)

    Beta = map(lambda o,p: -(o[0]*p[0]+ o[1]*p[1]),Omega,Points)
    #derive alpha by collocation
    N=[]
    for i in range(len(domain)):
        N_row = []
        for j in range(len(Omega)):
            z = Omega[j][0]*domain[i][0] + Omega[j][1]*domain[i][1]+Beta[j]
            N_row += [ BasisFunction.typeBasis(0,z,sigma)]
        N+=[N_row]
    N = numpy.matrix(N)
    T =numpy.matrix(target)
    T=T.T
    Alpha = list(N.I*T)
    Omega1 = map(lambda x:x[0],Omega)
    Omega2 = map(lambda x:x[1],Omega)

    #remove nodes with low alpha values
    threshold =0.01
    param = map(lambda a,b,o1,o2:[float(a),b,o1,o2], Alpha,Beta,Omega1,Omega2)
    param.sort(key=lambda x: x[0])
    i=0
    removals = 0
    while i != len(param):
        if abs(param[i][0]) < threshold:

            param=param[0:i]+param[i+1:]
            removals+=1
            i-=1
        i+=1
    param = numpy.array(param)
    param=param.flatten()

    


    
    return (param,removals)
        
