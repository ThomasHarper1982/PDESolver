##Solution class is an intermediate class between the neural network and the
##applications that use it. The solution representation is a function of
##the network term psi = f(N)
import random

from numpy import matrix
import numpy as np
from Am import *
from Network import *
from operator import add
from scipy import integrate
from ProblemInformation import *
from sympy import *
from McFall import *
from cmath import pi, exp
import random

class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

def randList2(n,m):

    result = []
    for i in range(n):
        row=[]
        for j in range(m):
            row+=[random.random()]
        result+=[row]
    return result
def differentiate(fn, x=Symbol('x'),xn=0,y=Symbol('y'), yn=0):
    d_fn = copy.deepcopy(fn)
    for i in range(xn):
        d_fn = diff(d_fn, x)
    for i in range(yn):
        d_fn = diff(d_fn, y)
    return d_fn
def matchBracket(s, i, bracket):
    if bracket == '(':
        lb = '('
        rb = ')'
    counter =1
    while counter > 0:
       # print s[i], counter
        if s[i] == lb:
            counter+=1
        elif s[i] == rb:
            counter-=1
        i+=1
    return i
def removeDerivativeString(d_str):
        l=d_str.find("D", 0)

        #print d_str
        while l != -1:       
            start_l = l
            comma_l = d_str.find(",", start_l)
            fn_str = d_str[start_l+1:comma_l]
            l+=2
            end_l = matchBracket(d_str, l, '(')
            x_count = 0
            y_count = 0
            term = d_str[start_l:end_l]
            p=0
            while p !=-1:
                p=term.find("x", p)

                if p !=-1:
                    x_count+=1
                    p+=1
            p=0
            while p !=-1:
                p=term.find("y", p)
                if p !=-1:
                    y_count+=1
                    p+=1   

            fn_str += "x"*x_count + "y"*y_count   
            d_str= d_str[0:start_l] + fn_str+ d_str[end_l-1:len(d_str)]
            l = d_str.find("D", 0)
        return d_str

class Solution:
    #holds all the terms in the problem, each as the same information except for DE and SF

     
    def McFallDirichletNetwork(xn, yn, X, fn_str, AdF=[], LdF=[]):
        '''
        For a paticular partial derivative:
        Evalutes the solution phi from the string fn_str for every X
        Given boundary satisfying function and length factor function values AdF and LdF
        '''
##        if xn==0 and yn==0:
        #print AdF
        S = Function('S(x,y)')
        Ad = Function('Ad(x,y)')
        Ld = Function('Ld(x,y)')
        x = Symbol('x')
        y= Symbol('y')
        N = (S(x,y)-Ad(x,y))/Ld(x,y)
        dN_str = str(differentiate(N,xn=xn,yn=yn)).replace('(x,y)',"").replace('(x, y)',"")
        dN_str = removeDerivativeString(dN_str)
        #print dN_str 
        x = X[0]
        y = X[1]
        S =    eval(str(differentiate(fn_str,xn=0,yn=0)))
        Sx =   eval(str(differentiate(fn_str,xn=1,yn=0)))
        Sxx =  eval(str(differentiate(fn_str,xn=2,yn=0)))
        Sy =   eval(str(differentiate(fn_str,xn=0,yn=1)))
        Syy =  eval(str(differentiate(fn_str,xn=0,yn=2)))
        Sxy =  eval(str(differentiate(fn_str,xn=1,yn=1)))
        Sxxx = eval(str(differentiate(fn_str,xn=3,yn=0)))
        Syyy = eval(str(differentiate(fn_str,xn=0,yn=3)))
        Sxxy = eval(str(differentiate(fn_str,xn=2,yn=1)))
        Sxyy = eval(str(differentiate(fn_str,xn=1,yn=2)))
        Ad = AdF[0][0]
        Ld = LdF[0][0]
        Adx = AdF[1][0]
        Ldx = LdF[1][0]
        Ady = AdF[0][1]
        Ldy = LdF[0][1]
        Adxx = AdF[2][0]
        Ldxx = LdF[2][0]
        Adyy = AdF[0][2]
        Ldyy = LdF[0][2]
        Adxy = AdF[1][1]
        Ldxy = LdF[1][1]
        Adxxx = AdF[3][0]
        Ldxxx = LdF[3][0]
        Adyyy = AdF[0][3]
        Ldyyy = LdF[0][3]
        Adxxy = AdF[2][1]
        Ldxxy = LdF[2][1]
        Adxyy = AdF[1][2]
        Ldxyy = LdF[1][2]
       # print S, Sx, Ad, Ld,Ldx

        if Ld>0.0000000001:
            N = eval(dN_str)
##            print float(S)
##            print re(float(S))
            return float(re(N))#re(float(S))
        else:
            return 0.0
    McFallDirichletNetwork = Callable(McFallDirichletNetwork)
                                            


    #combine the neumann and dirichlet components
    def McFallFunction(xn, yn, x,  parameters,AdF, LdF, gF, LmF, nx, ny,\
                       basis_type,param_derivative="",param_index=-1,don=[], mixed=False):
##    '''
##    xn - the x derivative
##    yn - the y derivative
##    x - the input vector
##    parameters - [alpha^i, beta^i, omega_1^i, omega_2^i], with i from 0 to n
##    AdF - a matrix of x,y partial derivatives - every derivative entry is the Dirichlet boundary satisfying function Ad
##    function values
##    Ldf - similarly a partial derivative matrix of Dirichlet length factor functions
##    gf - similarly a partial derivative matrix of Neumann boundary satisfying functions
##    LmF - simiarly a partial derivative matrix of Neumann length factor functions
##    basis_type - e.g. sigmoid
##    parameter_derivative - partially differentiates with respect to alpha, beta, omega1 or omega2
##    param_index - the node of the parameter being differentiated
##    '''   
        if xn==0 and yn==0:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Ad = AdF[0][0]
            Ld = LdF[0][0]
            am=0
            if mixed:
                Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
                Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
                am = Am(0,0,LdF,LmF,AdF,gF,nx,ny, [[N,Ny],[Nx]], bool(param_derivative),don=don)
##            print "Ad, am, Ld: ",Ad,am,Ld 
            S = Ad + am + Ld*N
        elif xn==1 and yn==0:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Adx = AdF[1][0]
            Ld = LdF[0][0]
            Ldx = LdF[1][0]
            am=0
            if mixed:
                Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
                am =Am(1,0,LdF,LmF,AdF,gF,nx,ny, [[N,Ny],[Nx,Nxy],[Nxx]], bool(param_derivative),don=don)
##            print "Adx, am, N, Ldx, Ld, Nx", Adx, am, N, Ldx, Ld, Nx
            S = Adx + am +N*Ldx  + Ld*Nx
        elif xn==2 and yn==0:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
            Adxx = AdF[2][0]
            Ld = LdF[0][0]
            Ldx = LdF[1][0]
            Ldxx = LdF[2][0]
            am=0
            if mixed:
                Nxxx = Network.function(3,0,param_derivative, x,parameters,basis_type, param_index)
                Nxxy = Network.function(2,1,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
                am =Am(2,0,LdF,LmF,AdF,gF,nx,ny, [[N,Ny],[Nx,Nxy],[Nxx,Nxxy],[Nxxx]], bool(param_derivative),don=don)
            S = Adxx +am + 2*Ldx*Nx + N*Ldxx + Ld*Nxx
        elif xn==3 and yn==0:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
            Nxxx = Network.function(3,0,param_derivative, x,parameters,basis_type, param_index)
            Adxxx = AdF[3][0]
            Ld = LdF[0][0]
            Ldx = LdF[1][0]
            Ldxx = LdF[2][0]
            Ldxxx = LdF[3][0]
            am=0
            if mixed:
                Nxxxx = Network.function(4,0,param_derivative, x,parameters,basis_type, param_index)
                Nxxxy = Network.function(3,1,param_derivative, x,parameters,basis_type, param_index)
                Nxxy = Network.function(2,1,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
                am = Am(3,0,LdF,LmF,AdF,gF,nx,ny, [[N,Ny],[Nx,Nxy],[Nxx,Nxxy],[Nxxx,Nxxxy],[Nxxxx]], bool(param_derivative),don=don)
            S = Adxxx + am +3*Nx*Ldxx + 3*Ldx*Nxx +  N*Ldxxx + Ld*Nxxx
        elif xn==1 and yn==1:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
            Adxy = AdF[1][1]
            Ld = LdF[0][0]
            Ldx = LdF[1][0]
            Ldxy = LdF[1][1]
            am=0
            if mixed:
                Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
                Nyy = Network.function(0,2,param_derivative, x,parameters,basis_type, param_index)
                Nxxy = Network.function(2,1,param_derivative, x,parameters,basis_type, param_index)
                Nxyy = Network.function(1,2,param_derivative, x,parameters,basis_type, param_index)
                am =Am(1,1,LdF,LmF,AdF,gF,nx,ny, [[N,Ny,Nyy],[Nx,Nxy,Nxyy],[Nxx,Nxxy]], bool(param_derivative),don=don)
            S = Adxy + am + Ny*Ldx + Ldy*Nx +  N*Ldxy + Ld*Nxy
        elif xn==0 and yn==1:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Ady = AdF[0][1]
            Ld = LdF[0][0]
            Ldy = LdF[0][1]
            am=0
            if mixed:
                Nyy = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
                am =Am(0,1,LdF,LmF,AdF,gF,nx,ny, [[N, Ny, Nyy],[Nx,Nxy]], bool(param_derivative),don=don)
                #print "Ady, am, N, Ldy, Ld, Ny", Ady, am, N, Ldy, Ld, Ny
            S = Ady + am + N*Ldy + Ld*Ny
        elif xn==0 and yn==2:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Nyy = Network.function(0,2,param_derivative, x,parameters,basis_type, param_index)
            Adyy = AdF[0][2]
            Ld = LdF[0][0]
            Ldy = LdF[0][1]
            Ldyy = LdF[0][2]
            am=0
            if mixed:
                Nyyy = Network.function(0,3,param_derivative, x,parameters,basis_type, param_index)
                Nxyy = Network.function(1,2,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
                am = Am(0,2,LdF,LmF,AdF,gF,nx,ny, [[N,Ny,Nyy,Nyyy],[Nx,Nxy,Nxyy]], bool(param_derivative),don=don)
            S = Adyy +am+ 2*Ldy*Ny + N*Ldyy + Ld*Nyy
        elif xn==0 and yn==3:
            N = Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Nyy = Network.function(0,2,param_derivative, x,parameters,basis_type, param_index)
            Nyyy = Network.function(0,3,param_derivative, x,parameters,basis_type, param_index)
            Adyyy = AdF[0][3]
            Ld = LdF[0][0]
            Ldy = LdF[0][1]
            Ldyy = LdF[0][2]
            Ldyyy = LdF[0][3]
            am=0
            if mixed:
                Nyyyy = Network.function(0,4,param_derivative, x,parameters,basis_type, param_index)
                Nxyyy = Network.function(1,3,param_derivative, x,parameters,basis_type, param_index)
                Nxyy = Network.function(1,2,param_derivative, x,parameters,basis_type, param_index)
                Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
                Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
                am =Am(0,3,LdF,LmF,AdF,gF,nx,ny, [[N,Ny,Nyy,Nyyy,Nyyyy],[Nx,Nxy,Nxyy,Nxyyy]], bool(param_derivative),don=don)
            S = Adyyy + am +3*Ny*Ldyy + 3*Ldy*Nyy +  N*Ldyyy + Ld*Nyyy  
        elif xn==2 and yn==1:
            N =Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
            Nxxy =Network.function(2,1,param_derivative, x,parameters,basis_type, param_index)
     
            Adxxy = AdF[2][1]
            Ld = LdF[0][0]
            Ldx = LdF[1][0]
            Ldxx =LdF[2][0]
            Ldy = LdF[0][1]
            Ldxy = LdF[1][1]
            Ldxxy = LdF[2][1]
            am=0
            if mixed:
                Nxxx = Network.function(3,0,param_derivative, x,parameters,basis_type, param_index)
                Nxxxy = Network.function(3,1,param_derivative, x,parameters,basis_type, param_index)
                Nyy = Network.function(0,2,param_derivative, x,parameters,basis_type, param_index)
                Nxyy = Network.function(1,2,param_derivative, x,parameters,basis_type, param_index)
                Nxxyy = Network.function(2,2,param_derivative, x,parameters,basis_type, param_index)
                am =Am(2,1,LdF,LmF,AdF,gF,nx,ny, [[N,Ny,Nyy],[Nx, Nxy, Nxyy],[Nxx, Nxxy,Nxxyy],[Nxxx,Nxxxy]], bool(param_derivative),don=don)
            S = Adxxy + am +2*Nx*Ldxy + 2*Ldx*Nxy + Ny*Ldxx + Ldy*Nxx +  N*Ldxxy + Ld*Nxxy
        elif xn==1 and yn==2:
            N =Network.function(0,0,param_derivative, x,parameters,basis_type, param_index)
            Ny = Network.function(0,1,param_derivative, x,parameters,basis_type, param_index)
            Nyy = Network.function(0,2,param_derivative, x,parameters,basis_type, param_index)
            Nx = Network.function(1,0,param_derivative, x,parameters,basis_type, param_index)
            Nxy = Network.function(1,1,param_derivative, x,parameters,basis_type, param_index)
            Nxyy =Network.function(1,2,param_derivative, x,parameters,basis_type, param_index)
     
            Adyyx = AdF[1][2]
            Ld = LdF[0][0]
            Ldy = LdF[0][1]
            Ldyy =LdF[0][2]
            Ldx = LdF[1][0]
            Ldxy = LdF[1][1]
            Ldyyx = LdF[1][2]
            am=0
            if mixed:
                Nyyy = Network.function(0,3,param_derivative, x,parameters,basis_type, param_index)
                Nxyyy = Network.function(1,3,param_derivative, x,parameters,basis_type, param_index)
                Nxx = Network.function(2,0,param_derivative, x,parameters,basis_type, param_index)
                Nxxy = Network.function(2,1,param_derivative, x,parameters,basis_type, param_index)
                Nxxyy = Network.function(2,2,param_derivative, x,parameters,basis_type, param_index)
                am =Am(1,2,LdF,LmF,AdF,gF,nx,ny, [[N,Ny,Nyy,Nyyy],[Nx, Nxy, Nxyy,Nxyyy],[Nxx, Nxxy,Nxxyy]], bool(param_derivative),don=don)
            S = Adyyx + am +2*Ny*Ldxy + 2*Ldy*Nxy + Nx*Ldyy + Ldx*Nyy + N*Ldyyx + Ld*Nyyx
        return float(S)
    McFallFunction = Callable(McFallFunction)
    
    def solution(xn, yn,i,j, x,  parameters, basis_type='sigmoid',representation='explicit',
                 param_derivative="",param_index=-1,mcfall=[], mixed=False):

        if representation == 'implicit':
            return Network.function(xn,yn,param_derivative, x,parameters,basis_type, param_index)
        elif representation == 'explicit':
            AdF = mcfall.Ad_values_fns[i][j]
            LdF = mcfall.LF_d[i][j]
            if mcfall.mixed:
                #print "mixed"
                LmF = mcfall.LF_m[i][j]
                gF = mcfall.g_values_fns[i][j]
                nx = mcfall.Normal_ox[i][j]
                ny = mcfall.Normal_oy[i][j]
            else:
                #print "not mixed"
                gF,LmF,nx,ny=[],[],[],[]    
            return Solution.McFallFunction(xn, yn,  x, parameters,AdF,LdF,gF,LmF,nx,ny,\
                                           basis_type, param_derivative=param_derivative,\
                                           param_index=param_index, don=[], mixed=mixed)

    solution = Callable(solution)
    def viewRequiredDirichletNetwork(xn,yn,prob,mcfall):
        xn,yn=0,0
        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle('Required Network with Dirichlet only conditions')
        #---- First subplot
        X,Y = np.meshgrid(prob.Xgrid[1:len(prob.Xgrid)-1],prob.Ygrid[1:len(prob.Ygrid)-1])
        k=3
        c=1
##        for i in range(4):
##            j=0
##            while j <=k:
##                if i<=xn and j<=yn:
##                    print xn+1,yn+1, c

                    #ax = fig.add_subplot(xn+1,yn+1, c, projection='3d')
        ax = fig.gca(projection='3d')
       # mpl.rcParams['legend.fontsize'] = 10
        #ax.axis('off')
        
        z = Solution.calculateNetwork2(xn,yn, prob.boundDomain, prob.sln, mcfall)
        Z =np.array(z)
        surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                               linewidth=0, antialiased=False)
        function_height = Z.max()-Z.min()
        ax.set_zlim3d(Z.min()-0.2*function_height,Z.max()+0.2*function_height)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(surf, shrink=0.5,aspect=5)
##                    c+=1
##                j+=1
##            k-=1
        plt.show()
    viewRequiredDirichletNetwork = Callable(viewRequiredDirichletNetwork)
    #the required network required to solve sln given a McFall Dirichlet Network
    def calculateNetwork2(xn, yn, domain, sln_str,mcfall):
        net = []
        for i in range(1,len(domain)-1):
            row=[]
            for j in range(1,len(domain[0])-1):
                #print i,j
                x=domain[i][j]
                row += [Solution.McFallDirichletNetwork(xn, yn, x, sln_str,
                                                        AdF=mcfall.Ad_values_fns[i][j], LdF=mcfall.LF_d[i][j])]
            net+=[row]
        return net
    calculateNetwork2 = Callable(calculateNetwork2)

    def viewSolution(xn, yn,domain,  parameters, basis_type='sigmoid',representation='explicit',\
                     param_derivative="",param_index=-1,mcfall=[], mixed=False, title ='Solution' ):

        

        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle(title)
        #---- First subplot
        print basis_type, representation
        X,Y = np.meshgrid(map(lambda x:x[0],domain[0]),map(lambda x:x[1],map(lambda x:x[0],domain)))
                          
        z = Solution.calculateSolutions2(xn, yn,domain,  parameters, basis_type=basis_type,representation=representation,\
                                         param_derivative=param_derivative,param_index=param_index,mcfall=mcfall, mixed=mixed)
        Z =np.array(z)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                               linewidth=0, antialiased=False)
        minV = Z.min()
        maxV = Z.max()
        function_height = maxV-minV
        ax.set_zlim3d(minV-0.2*function_height, maxV+0.2*function_height)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
        fig.colorbar(surf, shrink=0.5,aspect=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
    viewSolution = Callable(viewSolution)
    
    def calculateSolutions2(xn, yn,domain, parameters, basis_type='sigmoid',representation='explicit',\
                     param_derivative="",param_index=-1,mcfall=[], mixed=False):
        sln = []

        for i in range(len(domain)):
            row=[]
            for j in range(len(domain[0])):
               # print i,j, basis_type
                x=domain[i][j]
                row += [Solution.solution(xn, yn, i,j, x,  parameters, \
                                          basis_type=basis_type,representation=representation,\
                                          param_derivative=param_derivative,\
                                          param_index=param_index,mcfall=mcfall,mixed=mixed)]
            sln+=[row]
        return sln
    calculateSolutions2 = Callable(calculateSolutions2)


