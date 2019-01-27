
import math

class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class BasisFunction:
    '''Static class of basis function methods'''
       
    def Sigmoid(d,z):
        '''A sigmoid function takes the derivative order d and the input z'''
        z = max(z,-200)
        E = math.exp(-z);
        if 1+E>1.157920892373162e+76:
            E = 1.157920892373162e+76
        if d==0:
            y = 1/(1+E)
        elif d==1:
            y=E/pow((1+E),2)
        elif d==2:
            y = (pow(E,2)-E)/pow(1+E,3)
        elif d==3:       
            y = (E-4*pow(E,2) + 1*pow(E,3))/pow(1+E,4)
        elif d==4:
            y = (-E + 11*pow(E,2) -11*pow(E,3) +pow(E,4))/pow(1+E,5)
        elif d==5:
            y = (E - 26*pow(E,2) + 66*pow(E,3) -26*pow(E,4) + pow(E,5))/pow(1+E,6)
        elif d==6:
            y = (-E +57*pow(E,2) - 302*pow(E,3) +302*pow(E,4) -57*pow(E,5) + pow(E,6))/pow(1+E,7)
        elif d ==7:
            y = (E - 120*pow(E,2) + 1191*pow(E,3) -2416*pow(E,4) +1191*pow(E,5) -120*pow(E,6) + pow(E,7))/pow(1+E,8)
        else:
            print d      
        return y
    Sigmoid = Callable(Sigmoid)
    def Radial(d,z):
        '''A radial basis function takes the derivative order d and the input z'''
        z = max(z,-10)
        z = min(z,10)
        E = math.exp(-z**2);
        #return BasisFunction.Sigmoid(d+1,z)
        if d==0:    
            y = E
        if d==1:
            y=-2*z*E
        elif d==2:
            y = (4*z**2 - 2)*E
        elif d==3:
            y = (-8*z**3 + 12*z)*E
        elif d==4:
            y = (16*z**4 - 48*z**2 + 12)*E
        elif d==5:
            y = (-32*z**5 + 160*z**3 - 120*z)*E
        elif d==5:
            y = (64*z**6 - 480*z**4 + 720*z**2 - 120)*E
        elif d==7:
            y= (-128*z**7 + 1344*z**5 - 3360*z**3+ 1680*z)*E
        elif d==8:
            y=(256*z**8 - 3584*z**6 + 13440*z**4 - 13440*z**2 + 1680)*E
        return y
    Radial = Callable(Radial)
    def Sinusoid(d,z):
        '''A sinusoid takes the derivative order d and the input z'''
        d = d%4
        if d==0:
            try:
                y= math.sin(z)
            except:
                print z
        elif d==1:
            y= math.cos(z)
        elif d==2:
            y= -math.sin(z)
        elif d==3:
            y= -math.cos(z)
        return y
    Sinusoid = Callable(Sinusoid)   
    def Exp(d,z):
        y = ((2*math.pi*1j)**d)*math.exp(z*2*math.pi*1j)
        return y
    Exp = Callable(Exp) 
    def typeBasis(d,z,type_str):
        '''typeBasis takes the derivative order d and the input z and 
           the string literal:\nradial, sinusoid, sigmoid or  exp'''
        if type_str=='sigmoid':
            y = BasisFunction.Sigmoid(d,z)
        elif type_str=='radial':
            y = BasisFunction.Radial(d,z)
        elif type_str=='sinusoid':
            y = BasisFunction.Sinusoid(d,z)
        elif type_str=='exp':
            y = BasisFunction.Exp(d,z)
        return y
    typeBasis = Callable(typeBasis) 


        
        
          

