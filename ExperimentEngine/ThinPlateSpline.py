##ThinPlateSpline has been created and optimised to minimise repeated operations
##calculating partial derivatives - the inner part of the derivative is calculated once
##and so only the outer parts are calculated.



from numpy import matrix
import Functions
import math
from Functions import *
import numpy as np
import copy
def zeros(n,m):
    A=[]
    for i in range(n):
        A += [[0.0]*m]
    return matrix(A)

def TPSkernel(xn,yn,r):
    T=0
    if xn==0 and yn==0:
        T=r[0][0]**2*math.log(r[0][0]**2)
    elif xn==0 and yn==1:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[0][1] + 2*r[0][0]*r[0][1]
    elif xn==0 and yn==2:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[0][2] + 2*r[0][0]*r[0][2] + 2*math.log(r[0][0]**2)*r[0][1]**2 + 6*r[0][1]**2
    elif xn==0 and yn==3:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[0][3] + 2*r[0][0]*r[0][3] + 6*math.log(r[0][0]**2)*r[0][1]*r[0][2] + 18*r[0][1]*r[0][2] + 4*r[0][1]**3/r[0][0]
    elif xn==0 and yn==4:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[0][4] + 2*r[0][0]*r[0][4] + 8*math.log(r[0][0]**2)*r[0][1]*r[0][3] + 6*math.log(r[0][0]**2)*r[0][2]**2 + 24*r[0][1]*r[0][3] + 18*r[0][2]**2 + 24*r[0][1]**2*r[0][2]/r[0][0] - 4*r[0][1]**4/r[0][0]**2
    elif xn==1 and yn==0:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[1][0] + 2*r[0][0]*r[1][0]
    elif xn==1 and yn==1:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[1][1] + 2*r[0][0]*r[1][1] + 2*math.log(r[0][0]**2)*r[1][0]*r[0][1] + 6*r[1][0]*r[0][1]
    elif xn==1 and yn==2:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[1][2] + 2*r[0][0]*r[1][2] + 2*math.log(r[0][0]**2)*r[1][0]*r[0][2] + 4*math.log(r[0][0]**2)*r[0][1]*r[1][1] + 6*r[1][0]*r[0][2] + 12*r[0][1]*r[1][1] + 4*r[1][0]*r[0][1]**2/r[0][0]
    elif xn==1 and yn==3:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[1][3] + 2*r[0][0]*r[1][3] + 2*math.log(r[0][0]**2)*r[1][0]*r[0][3] + 6*math.log(r[0][0]**2)*r[0][1]*r[1][2] + 6*math.log(r[0][0]**2)*r[1][1]*r[0][2] + 6*r[1][0]*r[0][3] + 18*r[0][1]*r[1][2] + 18*r[1][1]*r[0][2] + 12*r[1][0]*r[0][1]*r[0][2]/r[0][0] + 12*r[0][1]**2*r[1][1]/r[0][0] - 4*r[1][0]*r[0][1]**3/r[0][0]**2
    elif xn==2 and yn==0:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[2][0] + 2*r[0][0]*r[2][0] + 2*math.log(r[0][0]**2)*r[1][0]**2 + 6*r[1][0]**2
    elif xn==2 and yn==1:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[2][1] + 2*r[0][0]*r[2][1] + 4*math.log(r[0][0]**2)*r[1][0]*r[1][1] + 2*math.log(r[0][0]**2)*r[0][1]*r[2][0] + 12*r[1][0]*r[1][1] + 6*r[0][1]*r[2][0] + 4*r[1][0]**2*r[0][1]/r[0][0]
    elif xn==2 and yn==2:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[2][2] + 2*r[0][0]*r[2][2] + 4*math.log(r[0][0]**2)*r[1][0]*r[1][2] + 4*math.log(r[0][0]**2)*r[0][1]*r[2][1] + 2*math.log(r[0][0]**2)*r[2][0]*r[0][2] + 4*math.log(r[0][0]**2)*r[1][1]**2 + 12*r[1][0]*r[1][2] + 12*r[0][1]*r[2][1] + 6*r[2][0]*r[0][2] + 12*r[1][1]**2 + 4*r[1][0]**2*r[0][2]/r[0][0] + 16*r[1][0]*r[0][1]*r[1][1]/r[0][0] + 4*r[0][1]**2*r[2][0]/r[0][0] - 4*r[1][0]**2*r[0][1]**2/r[0][0]**2
    elif xn==3 and yn==0:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[3][0] + 2*r[0][0]*r[3][0] + 6*math.log(r[0][0]**2)*r[1][0]*r[2][0] + 18*r[1][0]*r[2][0] + 4*r[1][0]**3/r[0][0]
    elif xn==3 and yn==1:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[3][1] + 2*r[0][0]*r[3][1] + 6*math.log(r[0][0]**2)*r[1][0]*r[2][1] + 2*math.log(r[0][0]**2)*r[0][1]*r[3][0] + 6*math.log(r[0][0]**2)*r[2][0]*r[1][1] + 18*r[1][0]*r[2][1] + 6*r[0][1]*r[3][0] + 18*r[2][0]*r[1][1] + 12*r[1][0]**2*r[1][1]/r[0][0] + 12*r[1][0]*r[0][1]*r[2][0]/r[0][0] - 4*r[1][0]**3*r[0][1]/r[0][0]**2
    elif xn==4 and yn==0:
        T=2*r[0][0]*math.log(r[0][0]**2)*r[4][0] + 2*r[0][0]*r[4][0] + 8*math.log(r[0][0]**2)*r[1][0]*r[3][0] + 6*math.log(r[0][0]**2)*r[2][0]**2 + 24*r[1][0]*r[3][0] + 18*r[2][0]**2 + 24*r[1][0]**2*r[2][0]/r[0][0] - 4*r[1][0]**4/r[0][0]**2
    return T

def EuclideanFn(xn,yn,x,x0,y,y0,d):
    E=0
    if xn==0 and yn==0:
        E=(d**2 + (x - x0)**2 + (y - y0)**2)**(1/2.0)
    elif xn==0 and yn==1:
        E=(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(1/2.0)
    elif xn==0 and yn==2:
        E=(-y + y0)*(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0) + (d**2 + (x - x0)**2 + (y - y0)**2)**(-1/2.0)
    elif xn==0 and yn==3:
        E=(-3*y + 3*y0)*(-y + y0)*(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) + 2*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0) - (y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==0 and yn==4:
        E=(-5*y + 5*y0)*(-3*y + 3*y0)*(-y + y0)*(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(7/2.0) + 3*(-3*y + 3*y0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 2*(-3*y + 3*y0)*(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 3*(-y + y0)*(y - y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 3/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==1 and yn==0:
        E=(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(1/2.0)
    elif xn==1 and yn==1:
        E=(x - x0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==1 and yn==2:
        E=(x - x0)*(-3*y + 3*y0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - (x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==1 and yn==3:
        E=(x - x0)*(-5*y + 5*y0)*(-3*y + 3*y0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(7/2.0) - 2*(x - x0)*(-3*y + 3*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 3*(x - x0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0)
    elif xn==2 and yn==0:
        E=(-x + x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0) + (d**2 + (x - x0)**2 + (y - y0)**2)**(-1/2.0)
    elif xn==2 and yn==1:
        E=(-x + x0)*(x - x0)*(-3*y + 3*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) + (-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==2 and yn==2:
        E=(-x + x0)*(x - x0)*(-5*y + 5*y0)*(-3*y + 3*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(7/2.0) - 3*(-x + x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) + (-3*y + 3*y0)*(-y + y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 1/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn==3 and yn==0:
        E=(-3*x + 3*x0)*(-x + x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) + 2*(-x + x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0) - (x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    elif xn== 3 and yn== 1:
        E=(-3*x + 3*x0)*(-x + x0)*(x - x0)*(-5*y + 5*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(7/2.0) + 2*(-x + x0)*(-3*y + 3*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - (x - x0)*(-3*y + 3*y0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0)
    elif xn== 4 and yn== 0:
        E=(-5*x + 5*x0)*(-3*x + 3*x0)*(-x + x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(7/2.0) + 3*(-3*x + 3*x0)*(-x + x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 2*(-3*x + 3*x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 3*(-x + x0)*(x - x0)/(d**2 + (x - x0)**2 + (y - y0)**2)**(5/2.0) - 3/(d**2 + (x - x0)**2 + (y - y0)**2)**(3/2.0)
    return E

class ThinPlateSpline:
    ##Performs the mapping from reference to target points
    def __init__(self, referencePoints, targetPoints,d=0.01):
        self.d = d
        self.referencePoints =copy.deepcopy(referencePoints)
        self.targetPoints = copy.deepcopy(targetPoints)
        self.num = len(referencePoints)
        self.M = self.num + 3;
        C = zeros(self.M, self.M);
        #print C, type(C)
        for i in range(self.num):
            for j in range(self.num):
                A=self.referencePoints[i]
                B=self.referencePoints[j]
                x = A[0]
                y= A[1]
                x0=B[0]
                y0=B[1]
                
                #print func["euclidean[0][0]"]
                #r =eval(func["euclidean[0][0]"])
                r=EuclideanFn(0,0,x,x0,y,y0,d)
                #print r, r1
                C[i,j] = (r**2)*math.log(r**2);
                #print x,y,x0,y0,r,C[i,j]
            C[i,j+1] = 1;
            C[i,j+2] = self.referencePoints[i][0]
            C[i,j+3] = self.referencePoints[i][1]
        C[self.num,0:self.num] = [1]*self.num
        C[self.num+1,0:self.num] = map(lambda x : x[0], self.referencePoints);
        C[self.num+2,0:self.num] = map(lambda x : x[1], self.referencePoints);
        
        RHS_X = matrix(map(lambda x : x[0], self.targetPoints) + [0 ,0 ,0]).T;
        RHS_Y = matrix(map(lambda x : x[1], self.targetPoints) + [0 ,0 ,0]).T;
        #need to find unknowns
       # print C
        #if the determinant of C is 0, then C has no inverse!
        self.F_X = C.I*RHS_X;
        self.F_Y = C.I*RHS_Y;
        self.trans = np.append(self.F_X,self.F_Y,1)

    def transformDomain(self,xn,yn,domain):
        fn=[]
        for i in range(len(domain)):
            row=[]
            for j in range(len(domain[0])):
                reference=domain[i][j]
                C=self.preProcessTransform(xn,yn,reference)
                row += [self.transformDerivatives(reference,C)]
            fn+=[row]
        return fn
    ##Euclidean values will be preprocessed once so it can be used in the outer function for multiple
    ##partial derivatives later
    def preProcessEuclidean(self,xn,yn,reference):
        Euclidean = []
        d=self.d
        #for each reference point, obtain euclidean derivative
        for l,ref in enumerate(self.referencePoints):
            #print l
            k=4
            E=[]
            for i in range(5):            
                row=[]
                j=0
                while j <= k:
                    if i<= xn and j <=yn:
                        
                        x0=ref[0]
                        y0=ref[1]
                        x=reference[0]
                        y=reference[1]
                        #E0=eval(func["euclidean["+str(i) +"]["+str(j)+"]"])
                        r=EuclideanFn(i,j,x,x0,y,y0,d)
                        #if xn+yn>0: 
                        #    print i,j#,E1-E0
                        row += [r]
                    else:
                        row += [0]
                    j+=1
                k-=1
                E+=[row]
            Euclidean+=[E]
        return Euclidean
    ##Taking the prepocessed Euclidean matrix (of partial derivatives)
    ##perform the transform for points on the domain return the matrix of constants
    def preProcessTransform(self,xn,yn,reference,Euclidean):
        d=self.d
        #C = zeros(1,self.M);
        C_fns=[]
        j=0
        #print Euclidean
        for l,ref in enumerate(self.referencePoints):
            k=4
            E=[]
            for i in range(5):            
                row=[]
                j=0
                while j <= k:
 
                    if i<= xn and j <=yn:
                        r = Euclidean[l]
                        #print r
                        #if i<= xn and j <=yn:
                        #T= eval(func["TPS_"+ str(i)+"_"+str(j)])
                        T = TPSkernel(i,j,r)
                        #if xn+yn>0:
                        #    print i,j,T-T1#func["TPS_"+ str(i)+"_"+str(j)]
                        #print i,j,"TPS_"+ str(i)+"_"+str(j),func["TPS_"+ str(i)+"_"+str(j)],r[0][0]
                        row+=[T]
                    else:
                        row += [0]      
                    j+=1
                E+=[row]
                k-=1
            C_fns+=[E]
        return C_fns

    ##Using the aformemetioned matrix of constants and the reference value (from the domain)
    ##calcuate a matrix of derivatives of the transform
    def transformDerivatives(self,xn,yn,reference,C_fns,dim=2):
        #print C_fns
        k=4
        D=[]
        for i in range(5):
            row=[]
            j=0
            while j <= k:
                if i<= xn and j <=yn:
                    C=map(lambda x: x[i][j], C_fns)+[0,0,0]
                    #C=map(lambda x: 0, C_fns)+[0,0,0]
                    if i+j>=2:
                        a,b,c=0,0,0
                    elif i>=1:
                        a,b,c=0,1,0
                    elif j>=1:
                        a,b,c=0,0,1
                    else:
                        a,b,c=1,reference[0],reference[1]
                    l= self.M-4
                    #a,b,c=0,0,0
                    #print j
                    C[l+1] = a
                    C[l+2] = b
                    C[l+3] = c

                    Target = matrix(C)*self.trans
                    if dim==2:
                        row+=[[Target[0,0],Target[0,1]]]
                    elif dim==1:
                        row+=[Target[0,0]]
                   # print i,j,Target
                else:
                    if dim==2:
                        row+=[[0,0]]
                    elif dim==1:
                        row+=[0]
                j+=1
            k-=1
            D+=[row]
        return D
        
    def transform(self,xn,yn,reference, C):
        C=map(lambda x: x[0][0], C)+[0,0,0]
        if xn+yn>=2:
            a,b,c=0,0,0
        elif xn>1:
            a,b,c=0,1,0
        elif yn>1:
            a,b,c=0,0,1
        else:
            a,b,c=1,reference[0],reference[1]
        j= self.M-4
        #print j
        C[j+1] = a
        C[j+2] = b
        C[j+3] = c

        Target = matrix(C)*self.trans
        return [Target[0,0],Target[0,1]]

