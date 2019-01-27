##A collection of methods dealing with Fourier Transforms
##and the inverse (with partial derivatives options xn, yn)

from scipy import *
from scipy.optimize import leastsq
import numpy
from numpy import matrix
from scipy import fftpack
import numpy as np
from cmath import pi, exp, sin, cos
from scipy import integrate
from sympy.utilities.iterables import flatten

def range2(a,b,step=1):
    new = []
    c=a
    k=abs(step)/step
    while c <= k*b:
        new+=[c]
        c+=step
    return new

def integrate(dY,X=[],h=1,f_0=0):
    Y = [f_0]
    if len(X)>1:
        h=X[1]-X[0]
    total=f_0
    #print h
    for i in range(0,len(dY)-1):
        total+=h*(dY[i]+dY[i+1])/2.0
        Y+=[total]
    #print Y
    return Y
    
def dft2(x,N1,N2):

    freq1 = []
    freq2 = []
    if N1%2==0:
        freq1 = range2(0, N1/2.0)+ range2(-N1/2+1,-1)
    else:
        freq1 = range2(0, N1/2.0)+ range2(-(N1/2),-1)
    if N2%2==0:
        freq2 = range2(0, N2/2.0)+ range2(-N2/2+1,-1)
    else:
        freq2 = range2(0, N2/2.0)+ range2(-(N2/2),-1)
    #print freq1,freq2
    Parameter = []
    for k1 in freq1:
        Row = []
        for k2 in freq2:
            output = 0#array([0,0])
            k = array([k1,k2])
            for n1 in range(0,N1):
                for n2 in range(0,N2):
                    n = array([float(n1)/N1,float(n2)/N2])
                   # print n1,n2,n,x[n1,n2],output
                    output +=  exp(-sum(k*n)*2*math.pi*1j)*x[n1,n2]# array([math.cos(sum(k*n)*2*math.pi), -math.sin(sum(k*n)*2*math.pi)])*x[n1,n2]
           # print output
            Row+= [output]
        Parameter+=[Row]
    Parameter = array(Parameter)
    return Parameter

def idft2_deriv(xn,yn,Parameter,N1,N2,Xaffine=[1,0],Yaffine=[1,0]):
    freq1 = []
    freq2 = []
    if N1%2==0:
        freq1 = range2(0, N1/2)+ range2(-(N1/2)+1,-1)
    else:
        freq1 = range2(0, N1/2)+ range2(-(N1/2),-1)
    if N2%2==0:
        freq2 = range2(0, N2/2)+ range2(-(N2/2)+1,-1)
    else:
        freq2 = range2(0, N2/2)+ range2(-(N2/2),-1)
    aff = float(Xaffine[0])
     
    #print "yn,xn",yn,xn,aff,N1,N2
    #print "freq", freq1, freq2, N1, N2
    Values = []
    for Y in range(0,N1):
        Row = []
        for X in range(0,N2):
            K=3
            dev=[]
            for i1 in range(4):
                dev_row=[]
                j1=0
                while j1 <= K:
                    if i1<= xn and j1 <=yn: 
                        #print i1,j1,X,Y
                        output = 0#array([0,0])
                        #n = array([Y/aff,X/aff])
                        n = array([Y,X])
                        #print Y,X,n
                        for a,k1 in enumerate(freq1) :
                            for b,k2 in enumerate(freq2):
                                k = array([float(k2)/N2,float(k1)/N1])
                                #if xn>0 or yn>0:
                                #    print a,b,k,(k[0]**xn)*(k[1]**yn)
                                #output += ((k[1])**xn)*((k[0])**yn)*((2*math.pi*aff*1j)**(xn+yn))*exp(sum(k*n)*2*aff*math.pi*1j)*Parameter[b,a]#array([math.cos(sum(k*n)*2*math.pi), math.sin(sum(k*n)*2*math.pi)])*Parameter[n1,n2]
                                output += ((k[1])**i1)*((k[0])**j1)*((2*math.pi*1j)**(i1+j1))*exp(sum(k*n)*2*math.pi*1j)*Parameter[b,a]#array([math.cos(sum(k*n)*2*math.pi), math.sin(sum(k*n)*2*math
                        dev_row+= [float(real((Xaffine[0]**i1)*(Yaffine[0]**j1)*output/(N1*N2)))]
                    else:
                        dev_row+=[0]
                    j1+=1
                K-=1
                dev+=[dev_row]
            Row += [dev]
        Values += [Row]
    #Values = array(Values)
    #print Values - Values.T
    return Values
    
        
def idft2(yn,xn,Parameter,N1,N2,Xaffine=[1,0],Yaffine=[1,0]):


    freq1 = []
    freq2 = []
    if N1%2==0:
        freq1 = range2(0, N1/2)+ range2(-(N1/2)+1,-1)
    else:
        freq1 = range2(0, N1/2)+ range2(-(N1/2),-1)
    if N2%2==0:
        freq2 = range2(0, N2/2)+ range2(-(N2/2)+1,-1)
    else:
        freq2 = range2(0, N2/2)+ range2(-(N2/2),-1)
    aff = float(Xaffine[0])
     
    #print "yn,xn",yn,xn,aff,N1,N2
    #print "freq", freq1, freq2, N1, N2
    Values = []
    for Y in range(0,N1):
        Row = []
        for X in range(0,N2):
            output = 0#array([0,0])
            #n = array([Y/aff,X/aff])
            n = array([Y,X])
            #print Y,X,n
            for a,k1 in enumerate(freq1) :
                for b,k2 in enumerate(freq2):
                    k = array([float(k2)/N2,float(k1)/N1])
                    #if xn>0 or yn>0:
                    #    print a,b,k,(k[0]**xn)*(k[1]**yn)
                    #output += ((k[1])**xn)*((k[0])**yn)*((2*math.pi*aff*1j)**(xn+yn))*exp(sum(k*n)*2*aff*math.pi*1j)*Parameter[b,a]#array([math.cos(sum(k*n)*2*math.pi), math.sin(sum(k*n)*2*math.pi)])*Parameter[n1,n2]
                    output += ((k[1])**xn)*((k[0])**yn)*((2*math.pi*1j)**(xn+yn))*exp(sum(k*n)*2*math.pi*1j)*Parameter[b,a]#array([math.cos(sum(k*n)*2*math.pi), math.sin(sum(k*n)*2*math
            Row+= [real((Xaffine[0]**xn)*(Yaffine[0]**yn)*output/(N1*N2))]
            #Row+= [output/(N1*N2)]
            #print Y,X, output/(N1*N2)
        Values+=[Row]
    Values = array(Values)
    #print Values - Values.T
    return Values


def Error(A,B):
    Am = matrix(A)
    Bm = matrix(B)
    return float((Am-Bm)*(Am-Bm).T)/len(Am[0,:])
def interpolateFunction(xn,yn,fn_values, domain):
##    print domain
    N1 = len(domain)
    N2 = len(domain[0])
    p1=domain[0][0]
    p2=domain[len(domain)-1][len(domain)-1]
    Xaffine    = matrix([[p1[0], 1],[p2[0], 1]]).I*matrix([0,N1-1]).T
    Yaffine    = matrix([[p1[1], 1],[p2[1], 1]]).I*matrix([0,N2-1]).T
    Xaffine = [float(Xaffine[0][0]),float(Xaffine[1][0])]
    Yaffine = [float(Yaffine[0][0]),float(Yaffine[1][0])]

    param = np.fft.fft2(fn_values)

    fns = idft2_deriv(xn,yn,param,N1,N2,Xaffine,Yaffine)

    print fns

                    
        
    return fns
    #derivative test, x axis

def meshgrid(X,Y):
##    A = []
##    for x in X:
##        row=[]
##        for y in Y:
##            row += [[x,y]]
##        A+=[row]
##    return array(A)
    boundDomain=[]
    for i in range(0,len(X)):
        boundDomain+=[map(lambda x: [x,Y[i]], X[0:len(X)])]
    return array(boundDomain)
def evaluateFunction(fn, grid):
    A = []
    #print range(len(grid)),range(len(grid[0:]))
    for i,y in enumerate(range(len(grid))):
        row=[]
        for j,x in enumerate(range(len(grid[0:]))):
            [x,y] = grid[i,j]
            #print fn,x,y,eval(fn)
            row+=[eval(fn)]
        A+=[row]
    return array(A)



