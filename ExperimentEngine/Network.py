import numpy
from numpy import matrix
from BasisFunction import BasisFunction
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class Network:
    def basis(d, Z, b):
        if b=='sigmoid':
            return BasisFunction.Sigmoid(d,Z)
        elif b=='radial':
            return BasisFunction.Radial(d,Z)
        elif b=='sinusoid':
            return BasisFunction.Sinusoid(d,Z)
        elif b=='exponential':
            return BasisFunction.Exp(d,Z)
    basis = Callable(basis)

    def no_derivative(nx,ny, x, parameters, basis_type):
        N=0
        X = matrix(x).T
        #print parameters
        for i in range(numpy.size(parameters)/4):
            Alpha = float(parameters[4*i])
            Beta = float(parameters[4*i+1])
            Omega = parameters[4*i+2:4*i+4]
            #print Alpha,Beta,Omega,X
            #print basis_type
            Z = float(Omega*X) + Beta
            Zx = float(Omega[0])
            Zy = float(Omega[1])
            N = N + Alpha*Network.basis(nx+ny, Z,basis_type)*(Zx**nx)*(Zy**ny)
        return N
    no_derivative = Callable(no_derivative)
    def omega_derivative(nx,ny,basis_index,omega_index,x, parameters, basis_type):
        N=0
        Alpha = parameters[4*basis_index]
        Beta = parameters[4*basis_index+1]
        Omega = parameters[4*basis_index+2:4*basis_index+4]
        X = matrix(x).T
        Z = float(Omega*X) + Beta
        Zo = x[omega_index]
        Zx =float(Omega[0])
        Zy = float(Omega[1])

        if omega_index == 0:
            Zxo = 1
            Zyo = 0
        else:
            Zxo = 0
            Zyo = 1
            
        if nx==0 and ny==0:       
            N = Alpha*Network.basis(0+0+1, Z,basis_type)*Zo
        elif nx==1 and ny==0:
            N = Alpha*Network.basis(1+0+1, Z,basis_type)*Zx*Zo +\
                Alpha*Network.basis(1+0+0, Z,basis_type)*Zxo
        elif nx==2 and ny==0:
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*(Zx**2)*Zo +\
                2*Alpha*Network.basis(2, Z,basis_type)*Zx*Zxo
        elif nx==0 and ny==1:
            N = Alpha*Network.basis(0+1+1, Z,basis_type)*Zy*Zo +\
                Alpha*Network.basis(0+0+1, Z,basis_type)*Zyo
        elif nx==0 and ny==2:
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*(Zy**2)*Zo +\
                2*Alpha*Network.basis(2, Z,basis_type)*Zy*Zyo
        elif nx==1 and ny==1:
            N = Alpha*Network.basis(3, Z,basis_type)*Zx*Zy*Zo +\
                Alpha*Network.basis(2, Z,basis_type)*Zx*Zyo +\
                Alpha*Network.basis(2, Z,basis_type)*Zy*Zxo
        elif nx==2 and ny==1:
            N = Alpha*Network.basis(4, Z,basis_type)*Zo*Zy*(Zx**2) + \
                Alpha*Network.basis(3, Z,basis_type)*Zyo*(Zx**2)\
                + 2*Alpha*Network.basis(3, Z,basis_type)*Zy*Zx*Zxo
        elif nx==1 and ny==2:
            N = Alpha*Network.basis(4, Z,basis_type)*Zo*Zx*(Zy**2) + \
                Alpha*Network.basis(3, Z,basis_type)*Zxo*(Zy**2)\
                + 2*Alpha*Network.basis(3, Z,basis_type)*Zx*Zy*Zyo   
        elif nx==3 and ny==0:
            N = Alpha*Network.basis(4,Z,basis_type)*Zo*(Zx**3) + \
                3*Alpha*Network.basis(3, Z,basis_type)*Zxo*(Zx**2)
        elif nx==0 and ny==3:
            N = Alpha*Network.basis(4,Z,basis_type)*Zo*(Zy**3) + \
                3*Alpha*Network.basis(3, Z,basis_type)*Zyo*(Zy**2)
        return N
    omega_derivative = Callable(omega_derivative)
    def alpha_derivative(nx,ny,basis_index,x, parameters, basis_type):
        N=0
        X = matrix(x).T
        Beta = parameters[4*basis_index+1]
        Omega = parameters[4*basis_index+2:4*basis_index+4]   
        X = matrix(x).T
        Z = float(Omega*X) + Beta
        Zx =float(Omega[0])
        Zy = float(Omega[1])
        N = Network.basis(nx+ny, Z, basis_type)*(Zx**nx)*(Zy**ny)
        return N
    
    alpha_derivative = Callable(alpha_derivative)
    def beta_derivative(nx,ny,basis_index,x, parameters, basis_type):
        N=0
        X = matrix(x).T
        Alpha = parameters[4*basis_index]
        Beta = parameters[4*basis_index+1]
        Omega = parameters[4*basis_index+2:4*basis_index+4]
        
        #print X
        Z = float(Omega*X) + Beta
        Zx = float(Omega[0])
        Zy = float(Omega[1])
        N = Alpha*Network.basis(nx+ny+1, Z,basis_type)*(Zx**nx)*(Zy**ny)
        return N
    
    beta_derivative = Callable(beta_derivative)
    def function(nx,ny,param_derivative, x, parameters, basis_type, basis_index=-1):
        if param_derivative=="":
            return Network.no_derivative(nx,ny, x, parameters, basis_type)
        elif param_derivative=="omega" or param_derivative=="omega1":
            return Network.omega_derivative(nx,ny,basis_index,0, x, parameters, basis_type)
        elif param_derivative=="omega2":
            return Network.omega_derivative(nx,ny,basis_index,1, x, parameters, basis_type)
        elif param_derivative=="alpha":
            #print nx,ny,param_derivative, x, parameters, basis_type, basis_index
            return Network.alpha_derivative(nx,ny,basis_index, x, parameters, basis_type)
        elif param_derivative=="beta":
            return Network.beta_derivative(nx,ny,basis_index, x, parameters, basis_type)
    function = Callable(function)

##P = matrix([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
##X = [1,1]
##
##print Network.function(0,0,"omega",X,P,'sigmoid',2)

