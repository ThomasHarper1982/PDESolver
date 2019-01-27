#Energy Function test
#function(nx,ny,param_derivative, x, parameters, basis_type, basis_index=-1)
import random
from numpy import matrix
import numpy as np
from Network import *
from operator import add
from scipy import integrate
from EnergyFunction import *
from ProblemInformation import *
import copy

def empLists(n,m):
    Lists = []
    for i in range(n):
        row = []
        Lists.append(row)
    return Lists

def randLists(n,m):
    Lists = []
    for i in range(n):
        row = []
        for j in range(m):
            row+=[5*random.random()]
        Lists.append(row)
    return Lists

def Error(A,B):
    Am = matrix(A)
    Bm = matrix(B)
    return float((Am-Bm)*(Am-Bm).T)/len(Am[0,:])

def associate(A,B):
    print "("+ str(A) +','+str(B)+" )"

##differential equation test
    
##basis = ["sinusoid"]
##param_deriv = ["alpha"]
##prob = ProblemInformation()
##prob.setDim(2)
##prob.setRepresention('implicit')
##prob.setDomain(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,1,10)))
##
###prob.setDifferentialEquation("x**2*F[0][0]-math.cos(x)*F[1][0]**2")
##prob.setDifferentialEquation("F[0][0]-F[1][0]")
##from scipy import integrate
##Lists = asarray(flatten(randLists(4,3)))
##Parameters = matrix(Lists)
###Parameters=matrix([[1000],[-math.log(1000)],[1],[1]])
###prob.setDomain(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,0,10)))
##prob.setDirichlet([[0,0]],[1])
###create inputs array

##basis = ["sigmoid","radial","sinusoid"]
##param_deriv = ["alpha","beta","omega","omega2"]
##
##
###basis = ["sinusoid"]
###param_deriv = ["alpha"]
##ParamLists=[]
##p_input = np.linspace(0,1,10)
##for j in range(4):
##    
##    paramTest = []
##    for i in range(10):
##        paramTest+=[copy.deepcopy(Lists)]
##        print j*len(paramTest[i])/4
##        paramTest[i][j*len(paramTest[i])/4 + 0] = p_input[i]
##     
##    ParamLists += [paramTest]
##
##for b in basis:
##    prob.setBasis(b)
##    for l,p in enumerate(param_deriv):
##        #print p
##        #print type(p),type(l)
##        Output_p =[]
##        Output =[]
##        #print len(ParamLists[l])
##        for param in ParamLists[l]:
##            #print param
##            Output_p +=[EnergyFunction.function(param,"DifferentialEquation",prob, p,0)] 
##            Output += [EnergyFunction.function(param,"DifferentialEquation",prob, "")]
##       # print len(Output_p),len(p_input)
##        #print Output_p,p_input
##        #print Output_p,Output
##        # print list(p_input), "***" ,Output_p[i][j]
##        
##        S_Output = integrate.cumtrapz(Output_p, list(p_input))
##        # print len(list(S_Output)),"****",len(Output[i][j])
##        #print Output[0], S_Output
##        S_Output=map(lambda x:x+Output[0], S_Output)
##        S_Output = [Output[0]] + S_Output
##        #print Output_p, Output, S_Output
##        #map(associate,list(S_Output),Output[i][j])
##        e = Error(list(S_Output),Output)
##        print b,p,e

#target mapping test
prob = ProblemInformation()
#prob.setDim(2)
prob.setRepresention('implicit')
prob.setTargetMapping(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,0,10)),numpy.linspace(0,1,10),2)
from scipy import integrate
Lists = asarray(flatten(randLists(4,3)))
#Parameters=matrix([[1000],[-math.log(1000)],[1],[1]])
#create inputs array

basis = ["sigmoid","radial","sinusoid"]
param_deriv = ["alpha","beta","omega","omega2"]

ParamLists=[]
p_input = np.linspace(0,1,10)
for j in range(4):
    
    paramTest = []
    for i in range(10):
        paramTest+=[copy.deepcopy(Lists)]
        #print j*len(paramTest[i])/4
        paramTest[i][j*len(paramTest[i])/4 + 0] = p_input[i]
     
    ParamLists += [paramTest]

for b in basis:
    prob.setBasis(b)
    for l,p in enumerate(param_deriv):
        #print p
        #print type(p),type(l)
        Output_p =[]
        Output =[]
        #print len(ParamLists[l])
        for param in ParamLists[l]:
            #print param
            Output_p +=[EnergyFunction.function(param,"TargetMapping",prob, p,0)] 
            Output += [EnergyFunction.function(param,"TargetMapping",prob, "")]
       # print len(Output_p),len(p_input)
        #print Output_p,p_input
        #print Output_p,Output
        # print list(p_input), "***" ,Output_p[i][j]
        
        S_Output = integrate.cumtrapz(Output_p, list(p_input))
        # print len(list(S_Output)),"****",len(Output[i][j])
        #print Output[0], S_Output
        S_Output=map(lambda x:x+Output[0], S_Output)
        S_Output = [Output[0]] + S_Output
        #print Output_p, Output, S_Output
        #map(associate,list(S_Output),Output[i][j])
        e = Error(list(S_Output),Output)
        print b,p,e
