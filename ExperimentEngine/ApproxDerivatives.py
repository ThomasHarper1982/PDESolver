import numpy
from numpy import matrix
from ThinPlateSpline import *
import math
import copy
import numpy as np
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import clock, time
##work out 2 dim forward derivatives, central differences
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
def integrate2(dY,X=[],h=1,f_0=0):
    Y = [f_0]
    if len(X)>1:
        h=X[1]-X[0]

    total=f_0

    for i in range(0,(dY.size-1)):
        total+=h*(dY[0,i]+dY[0,i+1])/2.0     
        Y+=[total]
    #print Y
    return Y

def localInterpolation(point, value, boundary_points, boundary_value,h,k):
    x=point[0]
    y=point[1]
    fn = 0
    omega = 10/h
    beta =  h/1.3
    diff=0
    for p,v in zip(boundary_points,boundary_value):
        x0 = p[0]
        y0 = p[1]
        if abs(x - x0)<0.00000001 and abs(y - y0)<0.00000001:
            #print "boundary ",x0,x,y0,y,v
            return v
        d= math.sqrt((x - x0)**2 + (y - y0)**2)
        diff += (v-value)/(1+math.exp(-omega*(-1*d +beta)))
    return value+diff

def interpolation(point, boundary_points, boundary_value,h,k,bound_weight=[]):
    x=point[0]
    y=point[1]
    fn = 0
   # print "\nbegin", point, "\n"
    #print zip(boundary_points,boundary_value)
    #print func["interpolant_"+str(xn)+"_"+str(yn)]
    i=0
    for p,v in zip(boundary_points,boundary_value):
        x0 = p[0]
        y0 = p[1] 
        if abs(x - x0)<0.00000001 and abs(y - y0)<0.00000001:
            #print "boundary ",x0,x,y0,y,v
            return v
        b=1
        if len(boundary_points)==len(bound_weight):
            b=bound_weight[i]
        #print v/math.sqrt((x - x0)**2.0 + (y - y0)**2.0)
        fn += b*v/(((x - x0)/h)**2.0 + ((y - y0)/k)**2.0)**1.0
        #fn += eval(func["interpolant_"+str(xn)+"_"+str(yn)])
        #print x,y,x0,y0,fn
   # print "end"
        i+=1
    return fn/len(boundary_points)

def interpolateDomain(domain_points, boundary_points, boundary_value,h,k,bound_weight=[]):
    domain_values = []
    #print domain_points
    h= domain_points[0][1][0]-domain_points[0][0][0]
    k= domain_points[1][0][1]-domain_points[0][0][1]
    #print h, k
    for i in range(len(domain_points)):
        row = []
        for j in range(len(domain_points[0])):
            
            d = domain_points[i][j]
            result = interpolation(d, boundary_points, boundary_value,h,k,bound_weight=bound_weight)
            #result = localInterpolation(d, result, boundary_points, boundary_value,h,k)
            #print d,interpolation(d, boundary_points, boundary_value)
            row += [result]
            #print interpolation(d, boundary_points, boundary_value),
            #print i,j,domain_points[i][j],interpolation(d, boundary_points, boundary_value,h,k)
        #print "\n"
        domain_values+=[row]
    return domain_values

#take a 3 by 3 subdomain and apply gaussian kernel
def window3by3(fn):
    total = 0
    sd=0.6
    for i,x in enumerate([1,0,1]):
        for j,y in enumerate([1,0,1]):
            total+=(1/(2*math.pi*(sd**2)))*math.exp(-(x+y)/(2*(sd**2)))*fn[i,j]
    return total

def smoothFunction(fn):
    #apply gaussian window to each domain point exluding the boundary,
    #stop at the middle 3 by 3
    #domain_mat = matrix(domain)
    fn_mat = matrix(fn)
    fn_mat_aux=copy.copy(fn_mat)
        #copy.copy(fn_mat)
    start = 1
    end = len(fn)-1
    while start+3 < end:
        for i in range(start,end):
            for j in range(start,end):
                 #print i,j,fn_mat[i-1:i+2,j-1:j+2]
                 fn_mat_aux[i,j]=window3by3(fn_mat[i-1:i+2,j-1:j+2])
        fn_mat = copy.copy(fn_mat_aux)
        start+=1
        end-=1
    fn=[]
    for i in range(fn_mat.shape[0]):
        row=[]
        for j in range(fn_mat.shape[1]):
            row+=[fn_mat[i,j]]
        fn+=[row]
        
    return fn

def Error(A,B):
    Am = matrix(A)
    Bm = matrix(B)
    return float((Am-Bm)*(Am-Bm).T)/len(Am[0,:])
def ErrorM(Am,Bm):
    return float((Am-Bm)*(Am-Bm).T)/len(Am[0,:])

#The Thin Plate Spline method of calculating boundary satisfying function interpolations#
def TPSBoundInterpolation(domain, boundary, target, xn=4, yn=4):
    print "TPS"
   # print target
    #since this TPS requires a 2d target, we must add a dummy dimension
    target = map(lambda x: [x,0], target)
   
    transform = ThinPlateSpline(boundary,target)

    TPS=[]
    time1=time()
    for i in range(len(domain)):
        TPS_row=[]
        for j in range(len(domain[0])):
            #print i,j
            E=transform.preProcessEuclidean(xn,yn,domain[i][j])
            C=transform.preProcessTransform(xn,yn,domain[i][j],E)
            TPS_row+=[transform.transformDerivatives(xn,yn,domain[i][j],C,dim=1)]
            #remove 
        TPS+=[TPS_row]
    if False:
        Derivatives=[]
        k=4
        for i in range(5):
            row=[]
            j=0
            while j <= k:
                Domain=[]
                for i0 in range(len(domain)):
                    domain_row=[]
                    for j0 in range(len(domain[0])):
                        domain_row+=[TPS[i0][j0][i][j]]
                    Domain+=[domain_row]
                fig = plt.figure()
                ax = fig.gca(projection='3d')
               # print "X,Y ",map(lambda x:x[0],domain[0]),map(lambda x:x[1],map(lambda x:x[0],domain))
                X,Y = np.meshgrid(map(lambda x:x[0],domain[0]),map(lambda x:x[1],map(lambda x:x[0],domain)))
                Z=np.array(Domain)
                #print X.shape, Y.shape,Z.shape
                surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                                       linewidth=0, antialiased=False)
                ax.set_zlim3d(0,1)
                ax.w_zaxis.set_major_locator(LinearLocator(10))
                ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

                fig.colorbar(surf, shrink=0.5,aspect=1)
                plt.show()
                #print i,j,"again"
                row+=[Domain]
                j+=1
            Derivatives+=[row]
            k-=1
    return TPS


#This class pertains to the methods with the Smoothing method - including interpolation of the function values
#while satisfying the boundary and calculating the derivatives using finite differences.
class approxFunction:

    def __init__(self, Xgrid, Ygrid, boundDomain, boundaryPoints, boundaryValues,\
                 divisions, interpolation_type = 'smoothing',xn_max=4,yn_max=4,bound_weight=[]):
        #self.newGrid(Xgrid, Ygrid, boundDomain)
        self.interpolation_type = interpolation_type
        self.xn_max=xn_max
        self.yn_max=yn_max
        self.Xgrid = Xgrid
        self.Ygrid = Ygrid
        self.hx = self.Xgrid[1]-self.Xgrid[0]
        self.hy = self.Ygrid[1]-self.Ygrid[0]
        self.boundDomain = boundDomain
        self.divisions = divisions
        self.newFunction(boundaryPoints, boundaryValues,self.hx,self.hy, divisions,bound_weight=bound_weight)

    def newFunction(self, boundaryPoints, boundaryValues,h,k, divisions,bound_weight=[]):
        if len(boundaryPoints)>0:
            if self.interpolation_type == 'smoothing':
                self.f0 = interpolateDomain(self.boundDomain, boundaryPoints, boundaryValues,h,k,bound_weight=bound_weight)
               # self.f = localInterpolation(self.boundDomain, boundaryPoints, boundaryValues,bound_weight=[])
                for i in range(50):
                    self.f0 = smoothFunction(self.f0)
                self.f0 = matrix(self.f0)
                if divisions >0:
                    self.f0=matrix(self.subdivide(self.boundDomain, self.f0, divisions))
                self.fns = self.calculateDerivatives([])
            elif self.interpolation_type == 'tps':
                self.fns = TPSBoundInterpolation(self.boundDomain, boundaryPoints, boundaryValues)            
        else:
            domain=[]
            for y in self.Ygrid:
                row=[]
                for x in self.Xgrid:
                    row+=[0]
                domain+=[row]
            self.f0 =domain
            self.f0 = matrix(self.f0)
            self.fns = self.calculateDerivatives([])
  
    def newGrid(self, Xgrid, Ygrid, boundDomain):
        self.density=5
        self.Xgrid = Xgrid
        self.Ygrid = Ygrid
        self.boundDomain = boundDomain
        self.newLen = self.density*len(Xgrid)
        
        self.NewXgrid = np.linspace(Xgrid[0],Xgrid[len(Xgrid)-1],self.density*(len(Xgrid)-1)+len(Xgrid))
        self.NewYgrid = np.linspace(Ygrid[0],Ygrid[len(Ygrid)-1],self.density*(len(Ygrid)-1)+len(Ygrid))
##        print self.Xgrid
##        print self.NewYgrid
##        self.hx = self.NewXgrid[1]-self.NewXgrid[0]
##        self.hy = self.NewYgrid[1]-self.NewYgrid[0]
        self.newBoundDomain = []
        for i in range(len(self.NewYgrid)):
            row=[]
            for j in range(len(self.NewXgrid)):
                row += [[self.NewXgrid[j],self.NewYgrid[i]]]
            self.newBoundDomain += [row]


    ##takes a rectangular mesh and increases the number of divisions, returns a new subdivided mesh
    def subdivide(self,domain_points, function_values, divisions):
        self.density=2**divisions
        self.NewXgrid = copy.deepcopy(self.Xgrid)
        self.NewYgrid = copy.deepcopy(self.Ygrid)
    #sub divide the domain
        for d in range(divisions):
##            print d
            self.NewXgrid = np.linspace(self.NewXgrid[0],self.NewXgrid[len(self.NewXgrid)-1],1*(len(self.NewXgrid)-1)+len(self.NewXgrid))
            self.NewYgrid = np.linspace(self.NewYgrid[0],self.NewYgrid[len(self.NewYgrid)-1],1*(len(self.NewYgrid)-1)+len(self.NewYgrid))
            print self.NewXgrid
            newBoundDomain=[]
            new_function_values=[]
            for i in range(len(self.NewYgrid)):
                row=[]
                row_f=[]
                for j in range(len(self.NewXgrid)):
                    if i%2==0 and j%2==0:
                        row_f+=[function_values[i/2][j/2]]
                    else:
                        row_f+=[0]
                    row += [[self.NewXgrid[j],self.NewYgrid[i]]]
                newBoundDomain += [row]
                new_function_values+=[row_f]

            for i in range(len(newBoundDomain)):
                for j in range(len(newBoundDomain[0])):
    ##                print i,j,new_function_values[i][j]
                    if i%2==1 and j%2==1:
                        new_function_values[i][j]=0.25*(new_function_values[i-1][j-1]+new_function_values[i+1][j-1]+\
                                                    new_function_values[i-1][j+1]+new_function_values[i+1][j+1])
                    elif i%2==1:
                        new_function_values[i][j]=0.5*(new_function_values[i-1][j]+new_function_values[i+1][j])
                    elif j%2==1:
                        new_function_values[i][j]=0.5*(new_function_values[i][j-1]+new_function_values[i][j+1])
            function_values = new_function_values

        self.new_hx = self.NewXgrid[1]-self.NewXgrid[0]
        self.new_hy = self.NewYgrid[1]-self.NewYgrid[0]
##        print newBoundDomain
##        print new_function_values        
        return   new_function_values

    
    def coefficients(self, dn, diffType,order):
        if diffType == 'central':
            if dn==0:
                s= [1]
            elif dn == 1:
                if order ==2:
                    s = [-1/2.0, 0, 1/2.0]
                elif order ==6:
                    s=[-1/60.0, 3/20.0, -3/4.0, 0, 3/4.0, -3/20.0, 1/60.0]
                elif order ==8:
                    s= [1/280.0, -4/105.0, 1/5.0, -4/5.0, 0.0, 4/5.0, -1/5.0, 4/105.0, -1/280.0]
            elif dn == 2:
                if order ==2:
                    s=[1,-2,1]
                elif order ==6:
                    s = [1/90.0, -3/20.0, 3/2.0, -49/18.0, 3/2.0, -3/20.0, 1/90.0]
                elif order ==8:
                    s= [-1/560.0, 8/315.0, -1/5.0, 8/5.0, -205/72.0, 8/5.0, -1/5.0, 8/315.0, -1/560.0]
            elif dn==3:
                if order==2:
                    s=[-1/2.0,1,0,-1,1/2.0]
                elif order==6:
                    s=[1/8.0, -1, 13/8.0, 0, -13/8.0, 1, -1/8.0]
                elif order ==8:
                    s= [-7/240.0, 3/10.0, -169/120.0, 61/30.0, 0.0, -61/30, 169/120, -3/10, 7/240]
            elif dn==4:
                if order==2:
                    s=[1,-4,6,-4,1]
                elif order==6:
                    s=[-1/6.0, 2.0, -13/2.0, 28/3.0, -13/2.0, 2.0, -1/6.0]
                elif order ==8:
                    s=[7/240.0,-2/5.0,169/60.0,-122/15.0, 91/8.0,-122/15.0,169/60.0,-2/5.0,7/240.0]
            return matrix(s)

        elif  diffType == 'forward' or diffType == 'backward':
            if dn==0:
                return [1]
            elif  dn == 1:
                if order ==2:
                    s=[-3/2.0,2,-1/2.0]
                    #s=[-1,1]
                elif order ==8:
                    s =[-49/20.0, 6.0, -15/2.0, 20/3.0, -15/4.0, 6/5.0, -1/6.0]
            elif dn == 2:
                if order ==2:
                    s=[2,-5,4,-1]
                elif order ==8:
                    s =[469/90.0, -223/10.0, 879/20.0, -949/18.0, 41.0, -201/10.0, 334/59.0, -7/10.0]
            elif dn == 3:
                if order ==2:
                    s=[-5/2.0, 9, -12, 7, -3/2.0]
                elif order ==8:
                    s =[-801/80.0, 349/6.0, -18353/120.0, 2391/10.0,-1457/6.0, 4891/30.0, -561/8.0, 527/30.0, -469/240.0]
            elif dn == 4:
                if order ==2:
                    s=[1,-4,6,-4,1]
                elif order ==8:
                    s=[1069/80.0,-1316/15.0,15289/60.0,-2144/5.0,10993/24.0,-4772/15.0,2803/20.0,-536/15.0,967/240.0]
            if diffType == 'backward':
                s=s[::-1]
                #print dn,(-1.0**(dn))
                s = ((-1.0)**(dn))*matrix(s)
                #s=-matrix(s)
            else:
                s = matrix(s)
        return s

    ##decide whether to employ central, backward or forward differences on the window of the domain 
    def fowardDifference(self,i0,j0,xn,yn,Xtype,Ytype,f,order):
        n=1
        #i0=self.density*i
        #j0=self.density*j
##        print "i0,j0, " ,i0, j0
##        print "start"
        if xn>0:
##            print "xderiv ", xn
            Cx = self.coefficients(xn,Xtype,order)
            if Xtype=='central':
                n=Cx[0,:].size/2 
                jp1= j0+n+1
                jm1= j0-n
            elif Xtype=='backward':
                #on the boundary
                n=Cx[0,:].size
                jp1 = j0+1
                jm1 = j0-n+1
            elif Xtype=='forward':
                n=Cx[0,:].size
##                print n
                jm1 = j0
                jp1 = j0+n
        else:
            Cx = matrix([1])
            jm1 = j0
            jp1 = j0+1
        if yn>0:
##            print "yderiv ", yn
##            print "y derivative"
            Cy = self.coefficients(yn,Ytype,order)   
            if Ytype=='central':
                n=Cy[0,:].size/2
                ip1= i0+n+1
                im1= i0-n
            elif Ytype=='backward':
                #on the boundary-1
                n=Cy[0,:].size
                ip1 = i0+1
                im1 = i0-n+1
            elif Ytype=='forward':
                
                n=Cy[0,:].size
##                print "n", n
                im1 = i0
                ip1 = i0+n    
        else:
            Cy = matrix([1])
            im1 = i0
            ip1 = i0+1

        F = f[im1:ip1,jm1:jp1]


        fd = Cy*F*Cx.T/((self.hx**xn)*(self.hy**yn))


       # print fd
        return float(fd[0][0])

    ##Calculation of derivatives on the interpolated function values (for the Smoothing method)
    def calculateDerivatives(self, F_list):
        xn_max=self.xn_max
        yn_max=self.yn_max
        Derivatives = []
        k=4
        for xn in range(5):
            deriv_row=[]
            yn=0
            while yn <= k:
                #print (xn,yn)
                #print xn,yn
                #if [i,j] in F_list:
                Domain=[]
                for i in range(len(self.Ygrid)):
                    domain_row = []
                    for j in range(len(self.Xgrid)):

                        if xn <= xn_max and yn <= yn_max: 
                            #deal with forward and central differences
                            Ytype = 'central'
                            Xtype = 'central'

                            #forward/backward on boundary, central elswehere
                            if i <2:
                                Ytype = 'forward'
                            elif i >= len(self.Ygrid)-2:
                                Ytype = 'backward'
                            if j <2:
                                Xtype = 'forward'
                            elif j >= len(self.Xgrid)-2:
                                Xtype = 'backward'

                            domain_row += [self.fowardDifference(i,j,xn,yn,Xtype,Ytype,self.f0,2)]
                        else:
                            domain_row +=[0]
                    Domain += [domain_row]
                yn+=1
                #print self.f[len(self.f)-1]
                #print Domain[8]
                deriv_row+=[Domain]

                if False:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    X,Y = np.meshgrid(self.Xgrid,self.Ygrid)
                    Z=np.array(Domain)
                    print X.shape, Y.shape,Z.shape
                    surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                                           linewidth=0, antialiased=False)
                    ax.set_zlim3d(1.2*Z.min(),1.2*Z.max())
                    ax.w_zaxis.set_major_locator(LinearLocator(10))
                    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

                    fig.colorbar(surf, shrink=0.5,aspect=1)
                    plt.show()

                
            k-=1
            Derivatives+=[deriv_row]

        Domain=[]
        for i in range(len(self.Ygrid)):
            row=[]
            for j in range(len(self.Xgrid)):
                aux_Derivatives=[]
                k=4
                for i0 in range(5):          
                    der_row=[]
                    j0=0
                    while j0 <= k:
                        der_row+=[Derivatives[i0][j0][i][j]]
                        j0+=1
                    k-=1
                    aux_Derivatives+=[der_row]
                row+=[aux_Derivatives]
            Domain+=[row]
        return Domain

    def calculateDerivatives2(self, F_list):
        Derivatives = []#copy.deepcopy(self.f)
        k=3
        for xn in range(4):
            deriv_row=[]
            yn=0
            while yn <= k:
                Domain=[]
                if xn>0 or yn>0:
                    print (xn,yn)
                    #print xn,yn
                    #if [i,j] in F_list:
                    for i in range(len(self.NewYgrid)):
                        domain_row = []
                        for j in range(len(self.NewXgrid)):
                            #print i,j
                            #deal with forward and central differences
                            Ytype= 'central'
                            Xtype = 'central'
                            if i <=4:
                                Ytype = 'forward'
                            elif i >= len(self.NewYgrid)-5:
                                Ytype = 'backward'
                            if j <=4:
                                Xtype = 'forward'
                            elif j >= len(self.NewXgrid)-5:
                                Xtype = 'backward'
    ##                        print Ytype,Xtype
                            xn2= xn
                            yn2= yn
                           # print xn2, yn2
                            #print Derivatives
                            if yn >= xn:
                                yn2 = yn-1
                                domain_row += [self.fowardDifference(i,j,0,1,Xtype,Ytype,deriv_row[yn2],8)]
                            elif yn < xn:
                                xn2 = xn-1
                                domain_row += [self.fowardDifference(i,j,1,0,Xtype,Ytype,Derivatives[xn2][yn2],8)]      
                        Domain += [domain_row]
                    
                    #print self.f[len(self.f)-1]
                    #print Domain[8]
                    deriv_row+=[matrix(Domain)]
                else:
                    #print self.f
                    deriv_row=[self.f]#[copy.deepcopy(self.f)]

                yn+=1
            k-=1
            Derivatives+=[deriv_row]
        Row=8
        Col=4
        print Derivatives[0][0][Row,:]
        print self.NewXgrid

        #test the derivatives via integration
        k=2
        for xn in range(3):
            deriv_row=[]
            yn=0
            while yn <= k:
                print xn,yn
                Fox=Derivatives[xn][yn][Row,:]
                Foxdx=Derivatives[xn+1][yn][Row,:]
                Fox_approx = matrix(integrate2(Foxdx, self.NewXgrid, f_0 = Fox[0,0])) 
                ex=ErrorM(Fox_approx,Fox)
                print Fox
                print Foxdx
                print Fox[0,0]
                print Fox_approx  
                print "error x", ex
                
                Foy=Derivatives[xn][yn][:,Col].T
                Foydy=Derivatives[xn][yn+1][:,Col].T
                Foy_approx = matrix(integrate2(Foydy, self.NewYgrid, f_0 = Foy[0,0]))
                ey=ErrorM(Foy_approx,Foy)


                yn+=1
            k-=1

   



