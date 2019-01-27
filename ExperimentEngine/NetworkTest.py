#network test
#function(nx,ny,param_derivative, x, parameters, basis_type, basis_index=-1)
import random
from numpy import matrix
import numpy as np
from Network import *
from operator import add
from scipy import integrate
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
            row+=[1*random.random()]
        Lists.append(row)
    return Lists

def Error(A,B):
    Am = matrix(A)
    Bm = matrix(B)
    return float((Am-Bm)*(Am-Bm).T)/len(Am[0,:])

def associate(A,B):
    print "("+ str(A) +','+str(B)+" )"
    
basis = ["sigmoid","radial","sinusoid"]
param_deriv = ["","alpha","beta","omega","omega2"]
##basis = ["sinusoid"]
##param_deriv = ["omega","omega2"]

Lists = randLists(4,10)
Parameters = matrix(Lists)

#create inputs array

##X = list(np.arange(0,1,1.0/99))
##Y = list(np.arange(0,1,1.0/99))
##
##XTest=[]
##for i in X:
##    XTest += [[i,1]]
##print len(XTest), len(X)
##YTest=[]
##    
##for j in Y:
##    YTest += [[1,j]]
###print XTest
##
##for b in basis:
##    for p in param_deriv:
##        Output =[]
##        k=3
##        for i in range(4):
##            col=[]
##            j=0
##            while j <= k:
##                print i,j
##                entry=[]
##                for l in XTest:
##                    entry += [Network.function(i,j,p,l,Parameters,b,0)]
##                col+=[entry]
##                j+=1
##            Output+=[col]
##            k-=1
##        k=2
##        for i in range(3):
##            j=0
##            while j <= k:
##                #print X, Output[i][j]
##                S_Output = integrate.cumtrapz(Output[i+1][j], X)
##               # print len(list(S_Output)),"****",len(Output[i][j])
##                S_Output=map(lambda x:x+Output[i][j][0], S_Output)
##                S_Output = [Output[i][j][0]] + S_Output
##                #map(associate,list(S_Output),Output[i][j])
##                e = Error(list(S_Output),Output[i][j])
##                print b,p,i,j,e
##                j+=1
##            k-=1
##print Parameters        
##for b in basis:
##    for p in param_deriv:
##        Output =[]
##        k=3
##        for i in range(4):
##            col=[]
##            j=0
##            while j <= k:
##                print j,i
##                entry=[]
##                for l in YTest:
##                    entry += [Network.function(i,j,p,l,Parameters,b,0)]
##                col+=[entry]
##                j+=1
##            Output+=[col]
##            k-=1
##        k=2
##        print len(Output), len(Output[0]), len(Output[1]), len(Output[2]), len(Output[3])
##        for i in range(3):
##            j=0
##            while j <= k:
##               # print X, Output[i][j]
##                S_Output = integrate.cumtrapz(Output[j][i+1], Y)
##               # print len(list(S_Output)),"****",len(Output[i][j])
##                S_Output=map(lambda x:x+Output[j][i][0], S_Output)
##                S_Output = [Output[j][i][0]] + S_Output
##                #map(associate,list(S_Output),Output[i][j])
##                e = Error(list(S_Output),Output[j][i])
##                print b,p,j,i,e
##                j+=1
##            k-=1
#integrate over parameters

basis = ["sigmoid"]
param_deriv = ["alpha","beta","omega","omega2"]
ParamLists=[]
p_input = np.linspace(-1,1,100)
for j in range(4):
    
    paramTest = []
    for i in range(100):
        paramTest+=[copy.deepcopy(Lists)]
        paramTest[i][j][0] = p_input[i]
     
    ParamLists += [paramTest]

x=[2,2]
for b in basis:
    for l,p in enumerate(param_deriv):
        #print l,len(ParamLists)
        #print type(p),type(l)
        Output_p =[]
        Output =[]
        k=3
        for i in range(4):
            col=[]
            col_p=[]
            j=0
            while j <= k:
                #print i,j
                entry_p=[]
                entry=[]
                for param in ParamLists[l]:
                    entry_p += [Network.function(i,j,p,x,matrix(param),b,0)]
                    entry += [Network.function(i,j,"",x,matrix(param),b,0)]
                col_p+=[entry_p]
                col+=[entry]
                j+=1
            Output_p+=[col_p]
            Output+=[col]
            k-=1
        k=2
       # print len(Output_p[0][0]),len(Output[0][0])
        for i in range(3):
            j=0
            while j <= k:
                #print Output_p[i][j], p_input
                S_Output = integrate.cumtrapz(Output_p[i][j], list(p_input))
               # print len(list(S_Output)),"****",len(Output[i][j])
                S_Output=map(lambda x:x+Output[i][j][0], S_Output)
                S_Output = [Output[i][j][0]] + S_Output
                #map(associate,list(S_Output),Output[i][j])
                e = Error(list(S_Output),Output[i][j])
                print b,p,i,j,e
                j+=1
            k-=1




