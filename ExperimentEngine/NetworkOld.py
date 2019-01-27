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
    basis = Callable(basis)

    def no_derivative(nx,ny, x, parameters, basis_type):
        N=0
        dim = len(x[0][0])
        if nx==0 and ny==0:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
               # print Omega
                #print X
                Z = float(Omega*X) + Beta
                N = N + Alpha*Network.basis(0+0, Z,basis_type)
            
        elif nx==1 and ny==0:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Z = float(Omega*X) + Beta
                Xx = matrix(x[1][0]).T
                Zx = float(Omega*Xx)
                N = N + Alpha*Network.basis(1+0, Z,basis_type)*Zx
            
        elif nx==2 and ny==0:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Z = float(Omega*X) + Beta
                Xx = matrix(x[1][0]).T
                Xxx = matrix(x[2][0]).T
                
                Zx = float(Omega*Xx)
                Zxx = float(Omega*Xxx)
                
                N = N + Alpha*Network.basis(2+0, Z,basis_type)*pow(Zx,2) + Alpha*Network.basis(1+0, Z,basis_type)*Zxx
        elif nx==3 and ny==0:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xx = matrix(x[1][0]).T
                Xxx = matrix(x[2][0]).T
                Xxxx = matrix(x[3][0]).T
                Z = float(Omega*X) + Beta
                Zx = float(Omega*Xx)
                Zxx = float(Omega*Xxx)         
                Zxxx = float(Omega*Xxxx)
                N = N + Alpha*Network.basis(3+0, Z,basis_type)*pow(Zx,3) + 3*Alpha*Network.basis(2+0, Z,basis_type)*Zx*Zxx + \
                    Alpha*Network.basis(1+0, Z,basis_type)*Zxxx

        elif nx==0 and ny==3:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xy = matrix(x[0][1]).T
                Xyy = matrix(x[0][2]).T
                Xyyy = matrix(x[0][3]).T                                     
                Z = float(Omega*X) + Beta
                Zy = float(Omega*Xy)
                Zyy = float(Omega*Xyy)          
                Zyyy = float(Omega*Xyyy)
                N = N + Alpha*Network.basis(3+0, Z,basis_type)*pow(Zy,3) + 3*Alpha*Network.basis(2+0, Z,basis_type)*Zy*Zyy + \
                    Alpha*Network.basis(1+0, Z,basis_type)*Zyyy

        elif nx==0 and ny==1:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xy = matrix(x[0][1]).T                                     
                Z = float(Omega*X) + Beta
                Zy = float(Omega*Xy)
                N = N + Alpha*Network.basis(0+1, Z,basis_type)*Zy
   
        elif nx==0 and ny==2:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xy = matrix(x[0][1]).T
                Xyy = matrix(x[0][2]).T                               
                Z = float(Omega*X) + Beta
                Zy = float(Omega*Xy)
                Zyy = float(Omega*Xyy)
                #print Alpha,Z, Zy, Zyy
                N = N + Alpha*Network.basis(2+0, Z,basis_type)*pow(Zy,2) + Alpha*Network.basis(1+0, Z,basis_type)*Zyy
     
        elif nx==1 and ny==1:
            for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xx = matrix(x[1][0]).T                                     
                Xy = matrix(x[0][1]).T
                Xxy = matrix(x[1][1]).T                                     
                Z =  float(Omega*X) + Beta
                Zx = float(Omega*Xx)
                Zy = float(Omega*Xy)
                Zxy = float(Omega*Xxy)
                N = N + Alpha*Network.basis(1+1, Z,basis_type)*Zx*Zy + Alpha*Network.basis(1+0, Z,basis_type)*Zxy
            
        elif nx==2 and ny==1:
           for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xx = matrix(x[1][0]).T                                     
                Xy = matrix(x[0][1]).T
                Xxy = matrix(x[1][1]).T
                Xxx = matrix(x[2][1]).T  
                Xxxy = matrix(x[2][1]).T                                     
                Z =  float(Omega*X) + Beta
                Zx = float(Omega*Xx)
                Zy = float(Omega*Xy)
                Zxy = float(Omega*Xxy)
                Zxx = float(Omega*Xxy)
                Zxxy = float(Omega*Xxxy)                                    
                N = N + Alpha*Network.basis(3+0, Z,basis_type)*Zy*pow(Zx,2) + 2*Alpha*Network.basis(2+0, Z,basis_type)*Zx*Zxy + \
                    Alpha*Network.basis(2+0, Z,basis_type)*Zy*Zxx + Alpha*Network.basis(1+0, Z,basis_type)*Zxxy

        elif nx==1 and ny==2:
           for i in range(len(parameters[1,:])):
                Alpha = float(parameters[0,i])
                Beta  = float(parameters[1,i])
                Omega = parameters[2:3+dim,i].T
                X = matrix(x[0][0]).T
                Xx = matrix(x[1][0]).T                                     
                Xy = matrix(x[0][1]).T
                Xxy = matrix(x[1][1]).T
                Xyy = matrix(x[0][2]).T                                    
                Xyyx = matrix(x[1][2]).T                                     
                Z =  float(Omega*X) + Beta
                Zx = float(Omega*Xx)
                Zy = float(Omega*Xy)
                Zxy = float(Omega*Xxy)
                Zyyx = float(Omega*Xyyx)
                Zyy = float(Omega*Xyy) 
                N = N + Alpha*Network.basis(3+0, Z,basis_type)*Zx*pow(Zy,2) + 2*Alpha*Network.basis(2+0, Z,basis_type)*Zy*Zxy + \
                    Alpha*Network.basis(2+0, Z,basis_type)*Zx*Zyy + Alpha*Network.basis(1+0, Z,basis_type)*Zyyx
        return N
    no_derivative = Callable(no_derivative)
    def omega_derivative(nx,ny,basis_index,omega_index,x, parameters, basis_type):
        N=0
        dim = len(x[0][0])
        if nx==0 and ny==0:       
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
           # print Omega
            #print X
            Z = float(Omega*X) + Beta
            Zo = x[0][0][omega_index]
            N = Alpha*Network.basis(0+0+1, Z,basis_type)*Zo
        elif nx==1 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta
            Zo = x[0][0][omega_index]
            Xx = matrix(x[1][0]).T
            Zx = float(Omega*Xx)
            Zxo = x[1][0][omega_index]
            N = Alpha*Network.basis(1+0+1, Z,basis_type)*Zx*Zo +\
                Alpha*Network.basis(1+0+0, Z,basis_type)*Zxo
        elif nx==2 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            Z = float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)
            Zo = x[0][0][omega_index]
            Zxo = x[1][0][omega_index]
            Zxxo = x[2][0][omega_index]
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*pow(Zx,2)*Zo +\
                2*Alpha*Network.basis(2, Z,basis_type)*pow(Zx,2)*Zx*Zxo +\
                Alpha*Network.basis(1+0+1, Z,basis_type)*Zxx*Zo +\
                Alpha*Network.basis(1, Z,basis_type)*Zxxo
        elif nx==0 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta
            Zo = x[0][0][omega_index]
            Xy = matrix(x[0][1]).T
            Zy = float(Omega*Xy)
            Zyo = x[0][1][omega_index]
            N = Alpha*Network.basis(0+1+1, Z,basis_type)*Zy*Zo +\
                Alpha*Network.basis(0+0+1, Z,basis_type)*Zyo
        elif nx==0 and ny==2:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)
            Zo = x[0][0][omega_index]
            Zyo = x[0][1][omega_index]
            Zyyo = x[0][2][omega_index]
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*pow(Zy,2)*Zo +\
                2*Alpha*Network.basis(2, Z,basis_type)*pow(Zy,2)*Zy*Zyo +\
                Alpha*Network.basis(1+0+1, Z,basis_type)*Zyy*Zo +\
                Alpha*Network.basis(1, Z,basis_type)*Zyyo
        
        elif nx==1 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zo = x[0][0][omega_index]
            Zyo = x[0][1][omega_index]
            Zxo = x[1][0][omega_index]
            Zxyo = x[1][1][omega_index]
            N = Alpha*Network.basis(3, Z,basis_type)*Zx*Zy*Zo +\
                Alpha*Network.basis(2, Z,basis_type)*Zyo*Zx +\
                Alpha*Network.basis(2, Z,basis_type)*Zy*Zxo +\
                Alpha*Network.basis(2+0, Z,basis_type)*Zo*Zxy+\
                Alpha*Network.basis(1+0, Z,basis_type)*Zxyo
  
        elif nx==2 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xxx = matrix(x[2][1]).T  
            Xxxy = matrix(x[2][1]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zxx = float(Omega*Xxy)
            Zxxy = float(Omega*Xxxy)
            Zo = x[0][0][omega_index]
            Zxo = x[1][0][omega_index]
            Zyo = x[0][1][omega_index]
            Zxxo = x[2][0][omega_index]
            Zxyo = x[1][1][omega_index]
            Zxxyo = x[2][1][omega_index]          
        
            N = Alpha*Network.basis(4, Z,basis_type)*Zo*Zy*(Zx**2) + \
                Alpha*Network.basis(3, Z,basis_type)*Zyo*(Zx**2)\
                + 2*Alpha*Network.basis(3, Z,basis_type)*Zy*Zx*Zxo +\
                2*Alpha*Network.basis(3, Z,basis_type)*Zo*Zx*Zxy\
                +2*Alpha*Network.basis(2,Z,basis_type)*Zxo*Zxy + \
                2*Alpha*Network.basis(2,Z,basis_type)*Zx*Zxyo\
                +Alpha*Network.basis(3, Z,basis_type)*Zo*Zy*Zxx +\
                Alpha*Network.basis(2,Z,basis_type)*Zyo*Zxx\
                +Alpha*Network.basis(2,Z,basis_type)*Zy*Zxxo + \
                Alpha*Network.basis(2,Z,basis_type)*Zo*Zxxy +\
                Alpha*Network.basis(1,Z,basis_type)*Zxxyo        
            
        elif nx==1 and ny==2:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xyy = matrix(x[1][2]).T  
            Xyyx = matrix(x[1][2]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zyy = float(Omega*Xyy)
            Zyyx = float(Omega*Xyyx)
            Zo = x[0][0][omega_index]
            Zxo = x[1][0][omega_index]
            Zyo = x[0][1][omega_index]
            Zyyo = x[0][2][omega_index]
            Zxyo = x[1][1][omega_index]
            Zyyxo = x[1][2][omega_index]          
        
            N = Alpha*Network.basis(4, Z,basis_type)*Zo*Zx*(Zy**2) + \
                Alpha*Network.basis(3, Z,basis_type)*Zxo*(Zy**2)\
                + 2*Alpha*Network.basis(3, Z,basis_type)*Zx*Zy*Zyo +\
                2*Alpha*Network.basis(3, Z,basis_type)*Zo*Zy*Zxy\
                +2*Alpha*Network.basis(2,Z,basis_type)*Zyo*Zxy + \
                2*Alpha*Network.basis(2,Z,basis_type)*Zy*Zxyo\
                +Alpha*Network.basis(3, Z,basis_type)*Zo*Zx*Zyy +\
                Alpha*Network.basis(2,Z,basis_type)*Zxo*Zyy\
                +Alpha*Network.basis(2,Z,basis_type)*Zx*Zyyo + \
                Alpha*Network.basis(2,Z,basis_type)*Zo*Zyyx +\
                Alpha*Network.basis(1,Z,basis_type)*Zyyxo  
            
        elif nx==3 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            Xxxx = matrix(x[3][0]).T
            Z = float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)         
            Zxxx = float(Omega*Xxxx)
            Zo = x[0][0][omega_index]
            Zxo = x[1][0][omega_index]
            Zxxo = x[2][0][omega_index]
            Zxxxo = x[3][0][omega_index]

            N = Alpha*Network.basis(4,Z,basis_type)*Zo*(Zx**3) + 3*Alpha*Network.basis(3, Z,basis_type)*(Zx**2)*Zxo +\
                3*Alpha*Network.basis(3, Z,basis_type)*Zo*Zx*Zxx + 3*Alpha*Network.basis(2, Z,basis_type)*Zxo*Zxx +\
                3*Alpha*Network.basis(2, Z,basis_type)*Zx*Zxxo + Alpha*Network.basis(2, Z,basis_type)*Zo*Zxxx +\
                Alpha*Network.basis(1, Z,basis_type)*Zxxxo    
        elif nx==0 and ny==3:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T
            Xyyy = matrix(x[0][3]).T
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)         
            Zyyy = float(Omega*Xyyy)
            Zo = x[0][0][omega_index]
            Zyo = x[0][1][omega_index]
            Zyyo = x[0][2][omega_index]
            Zyyyo = x[0][3][omega_index]

            N = Alpha*Network.basis(4,Z,basis_type)*Zo*(Zy**3) + 3*Alpha*Network.basis(3, Z,basis_type)*(Zy**2)*Zyo +\
                3*Alpha*Network.basis(3, Z,basis_type)*Zo*Zy*Zyy + 3*Alpha*Network.basis(2, Z,basis_type)*Zyo*Zyy +\
                3*Alpha*Network.basis(2, Z,basis_type)*Zy*Zyyo + Alpha*Network.basis(2, Z,basis_type)*Zo*Zyyy +\
                Alpha*Network.basis(1, Z,basis_type)*Zyyyo
        return N
    omega_derivative = Callable(omega_derivative)
    def alpha_derivative(nx,ny,basis_index,x, parameters, basis_type):
        N=0
        dim = len(x[0][0])
        if nx==0 and ny==0 :       
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
           # print Omega
            #print X
            Z = float(Omega*X) + Beta
            N = Network.basis(0+0, Z, basis_type)
        elif nx==1 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta
            Xx = matrix(x[1][0]).T
            Zx = float(Omega*Xx)
            N = Alpha*Network.basis(1+0, Z,basis_type)*Zx
            
        elif nx==2 and ny==0:
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)
            
            N = Network.basis(2+0, Z,basis_type)*pow(Zx,2) + Network.basis(1+0, Z,basis_type)*Zxx

        elif nx==0 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T                                     
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            N = Network.basis(0+1, Z,basis_type)*Zy
        elif nx==0 and ny==2:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T                               
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)
            #print Alpha,Z, Zy, Zyy
            N = Network.basis(2+0, Z,basis_type)*pow(Zy,2) + Network.basis(1+0, Z,basis_type)*Zyy

        elif nx==1 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            N = Network.basis(1+1, Z,basis_type)*Zx*Zy + Network.basis(1+0, Z,basis_type)*Zxy

        elif nx==2 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xxx = matrix(x[2][1]).T  
            Xxxy = matrix(x[2][1]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zxx = float(Omega*Xxy)
            Zxxy = float(Omega*Xxxy)                                    
            N = N + Network.basis(3+0, Z,basis_type)*Zy*pow(Zx,2) +\
                2*Network.basis(2+0, Z,basis_type)*Zx*Zxy + \
                Network.basis(2+0, Z,basis_type)*Zy*Zxx +\
                Network.basis(1+0, Z,basis_type)*Zxxy

        elif nx==1 and ny==2:
            Alpha = float(parameters[0,i])
            Beta  = float(parameters[1,i])
            Omega = parameters[2:3+dim,i].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xyy = matrix(x[0][2]).T                                    
            Xyyx = matrix(x[1][2]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zyyx = float(Omega*Xyyx)
            Zyy = float(Omega*Xyy) 
            N = Network.basis(3+0, Z,basis_type)*Zx*pow(Zy,2) +\
                2*Network.basis(2+0, Z,basis_type)*Zy*Zxy + \
                Network.basis(2+0, Z,basis_type)*Zx*Zyy +\
                Network.basis(1+0, Z,basis_type)*Zyyx

        elif nx==3 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            Xxxx = matrix(x[2][0]).T
            Z = float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)         
            Zxxx = float(Omega*Xxxx)
            N = N + Network.basis(3+0, Z,basis_type)*pow(Zx,3) +\
                3*Network.basis(2+0, Z,basis_type)*Zx*Zxx +\
                Network.basis(1+0, Z,basis_type)*Zxxx
        elif nx==0 and ny==3:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T
            Xyyy = matrix(x[0][3]).T                                     
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)          
            Zyyy = float(Omega*Xyyy)
            N = N + Network.basis(3+0, Z,basis_type)*pow(Zy,3) + 3*Network.basis(2+0, Z,basis_type)*Zy*Zyy + \
                Network.basis(1+0, Z,basis_type)*Zyyy
            
        return N
    alpha_derivative = Callable(alpha_derivative)
    def beta_derivative(nx,ny,basis_index,x, parameters, basis_type):
        N=0
        dim = len(x[0][0])
        if nx==0 and ny==0:       

            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            # print Omega
            #print X
            Z = float(Omega*X) + Beta
            Zb = 1
            N = Alpha*Network.basis(0+0+1, Z,basis_type)*Zb 
        elif nx==1 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta 
            Xx = matrix(x[1][0]).T
            Zx = float(Omega*Xx)
            Zb = 1
            N = Alpha*Network.basis(1+0+1, Z,basis_type)*Zx*Zb
        elif nx==2 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Z = float(Omega*X) + Beta
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)
            Zb = 1
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*(Zx**2)*Zb + Alpha*Network.basis(1+0+1, Z,basis_type)*Zxx*Zb
        elif nx==0 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Zy = float(Omega*Xy)
            Z = float(Omega*X) + Beta
            Zb = 1
            N = Alpha*Network.basis(1+0+1, Z,basis_type)*Zy*Zb
        elif nx==0 and ny==2:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T                               
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)
            Zb = 1
            N = Alpha*Network.basis(2+0+1, Z,basis_type)*(Zy**2)*Zb + Alpha*Network.basis(1+0+1, Z,basis_type)*Zyy*Zb
        elif nx==1 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zb = 1
            N = Alpha*Network.basis(3,Z,basis_type)*Zx*Zy*Zb +Alpha*Network.basis(2,Z,basis_type)*Zb*Zxy
        elif nx==2 and ny==1:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xxx = matrix(x[2][1]).T  
            Xxxy = matrix(x[2][1]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zxx = float(Omega*Xxy)
            Zxxy = float(Omega*Xxxy) 
            Zb = 1
            N = Alpha*Network.basis(4,Z,basis_type)*(Zx**2)*Zy*Zb + 2*Alpha*Network.basis(3,Z,basis_type)*Zxy*Zx*Zb +\
                Alpha*Network.basis(3,Z,basis_type)*Zxx*Zy*Zb +  Alpha*Network.basis(2,Z,basis_type)*Zxxy*Zb
        elif nx==1 and ny==2:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T                                     
            Xy = matrix(x[0][1]).T
            Xxy = matrix(x[1][1]).T
            Xyy = matrix(x[0][2]).T                                    
            Xyyx = matrix(x[1][2]).T                                     
            Z =  float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zy = float(Omega*Xy)
            Zxy = float(Omega*Xxy)
            Zyyx = float(Omega*Xyyx)
            Zyy = float(Omega*Xyy) 
            Zb = 1
            N = Alpha*Network.basis(4,Z,basis_type)*(Zy**2)*Zx*Zb + 2*Alpha*Network.basis(3,Z,basis_type)*Zxy*Zy*Zb +\
                Alpha*Network.basis(3,Z,basis_type)*Zyy*Zx*Zb +  Alpha*Network.basis(2,Z,basis_type)*Zyyx*Zb       
        elif nx==3 and ny==0:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xx = matrix(x[1][0]).T
            Xxx = matrix(x[2][0]).T
            Xxxx = matrix(x[2][0]).T
            Z = float(Omega*X) + Beta
            Zx = float(Omega*Xx)
            Zxx = float(Omega*Xxx)         
            Zxxx = float(Omega*Xxxx)
            Zb = 1
            N = Alpha*Network.basis(4,Z,basis_type)*Zb*Zx**3 + 3*Alpha*Network.basis(3,Z,basis_type)*Zb*Zx*Zxx + \
                Alpha*Network.basis(2,Z,basis_type)*Zb*Zxxx
        elif nx==0 and ny==3:
            Alpha = float(parameters[0,basis_index])
            Beta  = float(parameters[1,basis_index])
            Omega = parameters[2:3+dim,basis_index].T
            X = matrix(x[0][0]).T
            Xy = matrix(x[0][1]).T
            Xyy = matrix(x[0][2]).T
            Xyyy = matrix(x[0][3]).T                                     
            Z = float(Omega*X) + Beta
            Zy = float(Omega*Xy)
            Zyy = float(Omega*Xyy)          
            Zyyy = float(Omega*Xyyy)
            Zb = 1
            N = Alpha*Network.basis(4,Z,basis_type)*Zb*Zy**3 + 3*Alpha*Network.basis(3,Z,basis_type)*Zb*Zy*Zyy + \
                Alpha*Network.basis(2,Z,basis_type)*Zb*Zyyy   
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
            return Network.alpha_derivative(nx,ny,basis_index, x, parameters, basis_type)
        elif param_derivative=="beta":
            return Network.beta_derivative(nx,ny,basis_index, x, parameters, basis_type)
    function = Callable(function)

P = matrix([[1],[2],[3],[4]])
X = [[[1,1],[1,0],[0,0],[0,0]],[[0,1],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0]]]

print Network.function(3,0,"omega2",X,P,'sigmoid',0)

