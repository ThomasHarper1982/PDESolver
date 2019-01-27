import numpy
from numpy import atleast_1d, eye, mgrid, argmin, zeros, shape, \
     squeeze, vectorize, asarray, absolute, sqrt, Inf, asfarray, isinf

from scipy.optimize.linesearch import *
from scipy.optimize.linesearch import line_search_BFGS
from optimize import fmin_bfgs
from optimize import fmin
def rosen(x):
    """The Rosenbrock function"""
##    y = 0
##    for i in range(1, len(x)):
##        y += 100*(x[i] - x[i-1]**2)**2 + (1 - x[i-1])**2
##    return y
    x = asarray(x)
    return numpy.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0,axis=0)
def rosen_der(x):
    """The Rosenbrock function"""
    
##    der_y = [0]*len(x)
##    der_y[0] = -400*x[0]*(x[1] - x[0]**2)-2*(1-x[0])
##    
##    for j in range(1, len(x)-1):
##        # print j,x[j]
##         der_y[j] = 200*(x[j] - x[j-1]**2) -400*(x[j+1]-x[j]) - 2*(1 - x[j-1])
##         
##    der_y[len(x)-1] = 200*(x[len(x)-1] - x[len(x)-2]**2)
##    der_y = asarray(der_y)
##    return der_y
    x = asarray(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = numpy.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
x0 = [0.8,1.2,0.7,0.3,0.2]
x = fmin(rosen,x0)
#x = fmin_bfgs(rosen, x0, fprime=rosen_der, maxiter=10)
print x

#from EnergyFunction import *
##differential equation, 2d
##prob = ProblemInformation()
##prob.setDim(2)
##prob.setBasis('sigmoid')
##prob.setRepresention('implicit')
##param=matrix([5,2,4,5,3,2,6,2,3,1,2,3])
##prob.setDomain(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,0,10)))
##prob.setDifferentialEquation("F[0][0]-F[1][0]")
##prob.setDirichlet([[0,0]],[1])
##print "begin"
##x = fmin_bfgs(EnergyFunction.function, param, fprime=EnergyFunction.jacobian, maxiter=80, args=["DifferentialEquation",prob])

#differential equation, 1d
##prob = ProblemInformation()
##prob.setDim(1)
##prob.setBasis('sinusoid')
##prob.setRepresention('implicit')
##param=matrix([5,2,4,5,3,2,6,2,3])
##prob.setDomain(map(lambda x:[x],numpy.linspace(0,1,10)))
##prob.setDifferentialEquation("F[0]-F[1]")
##prob.setDirichlet([[0]],[1])
##print "begin"
##x = fmin_bfgs(EnergyFunction.function, param, fprime=EnergyFunction.jacobian, maxiter=80, args=["DifferentialEquation",prob])
##target mapping, 2d
##prob = ProblemInformation()
##prob.setDim(2)
##prob.setBasis('sigmoid')
##prob.setRepresention('implicit')
##param=matrix([5,2,4,5,3,2,6,2,3,1,2,3])
##prob.setTargetMapping(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,0,10)),\
##                      map(lambda x: math.exp(x), numpy.linspace(0,1,10)), 2)
##print "begin"
##x = fmin_bfgs(EnergyFunction.function, param, fprime=EnergyFunction.jacobian, maxiter=80,\
##                      args=["TargetMapping",prob])
##print x
##prob = ProblemInformation()
##prob.setDim(1)
##prob.setBasis('sigmoid')
##prob.setRepresention('implicit')
##param=matrix([5,2,4,5,3,2,6,2,3])
##prob.setTargetMapping(map(lambda x : [x],numpy.linspace(0,1,10)), map(lambda x: math.exp(x), numpy.linspace(0,1,10)), 1)
##print "begin"
##x = fmin_bfgs(EnergyFunction.function, param, fprime=EnergyFunction.jacobian, maxiter=80,\
##                      args=["TargetMapping",prob])

##print x
