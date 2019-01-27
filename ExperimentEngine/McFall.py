import copy
from ThinPlateSpline import *
from ProblemInformation import *
from ApproxDerivatives import *
from FourierTransform import *
import numpy
from numpy import matrix
import numpy as np
from Functions import *
from time import clock, time
import os

def determineNormals(boundary_points):
    '''Take a set of boundary points, determine the direction between boundary points
       and find the orthogonal direction to obtain the normal'''
    Normals = []
    for i in range(len(boundary_points)):
        #find the direction
        #print boundary_points[(i+1)%len(boundary_points)], matrix([1.0512279261376849, 0.52309697059073945])
        next_point = matrix(boundary_points[(i+1)%len(boundary_points)])
        this_point = matrix(boundary_points[i%len(boundary_points)])
        next_direction = next_point-this_point;
        next_direction = next_direction/math.sqrt(next_point*next_point.T);
        #work out normals
        n= matrix([[0,-1],[1,0]])*next_direction.T;
        Normals+=[[float(n[0]),float(n[1])]]
        #decide, should the normals always point in the positive directions?
        #last_normal = [last_direction(2), -last_direction(1)];
    return Normals       

def interpolation(point, boundary_points, boundary_value,h,k):
    '''For a point, and a set of corresponding boundary points and values,
        take a weighted eucidean sum between the point and the boundary points,
        return the boundary interpolated value at the point'''
    x=point[0]
    y=point[1]
    fn = 0
   # print "\nbegin", point, "\n"
    #print zip(boundary_points,boundary_value)
    #print func["interpolant_"+str(xn)+"_"+str(yn)]
    for p,v in zip(boundary_points,boundary_value):
        x0 = p[0]
        y0 = p[1] 
        if abs(x - x0)<0.00000001 and abs(y - y0)<0.00000001:
            #print "boundary ",x0,x,y0,y,v
            return v
        #print v/math.sqrt((x - x0)**2.0 + (y - y0)**2.0)
        fn += v/math.sqrt(((x - x0)**2.0)/h + ((y - y0)**2.0)/k)
        #fn += eval(func["interpolant_"+str(xn)+"_"+str(yn)])
        #print x,y,x0,y0,fn
   # print "end"    
    return fn/len(boundary_points)

def interpolateDomain(domain_points, boundary_points, boundary_value,h,k):
    '''For a rectangular set of domain points, use the corresponding boundary
       values and points to interpolate the boundary function at domain points'''
    domain_values = []
   
    for i in range(len(domain_points)):
        row = []
        for j in range(len(domain_points[0])):
            d = domain_points[i][j]
            #print d,interpolation(d, boundary_points, boundary_value)
            row += [interpolation(d, boundary_points, boundary_value,h,k)]
            #print i,j,d,row[j]
        domain_values+=[row]
    return domain_values
def newFunction(boundDomain,boundaryPoints, boundaryValues,h,k):
    '''Interpolate the domain points, then apply a smoothing function to smooth the
       interpolated domain points until convergence'''
    f = interpolateDomain(boundDomain, boundaryPoints, boundaryValues,h,k)
    #self.f = localInterpolation(self.boundDomain, boundaryPoints, boundaryValues)
    print len(f), len(f[0])
    for i in range(500):
       # print i
        f = smoothFunction(f)
    print len(f), len(f[0])    
    return f

        #self.f = matrix(self.f)
#take a 3 by 3 subdomain and apply gaussian kernel
def window3by3(fn):
    '''Take the 3 by 3 rectangular function values and apply a gaussian filter'''
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
    """
    apply the gaussian window to each domain point besides the boundary
    """
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
def integrate(dY,X=[],h=1,f_0=0):
        Y = [f_0]
        if len(X)>1:
            h=X[1]-X[0]
        total=f_0
        #print h
        for i in range(0,len(dY)-1):
            total+=h*(dY[i]+dY[i+1])/2.0
            Y+=[total]
        return Y
        #print Y
def Error(A,B):
        Am = matrix(A)
        Bm = matrix(B)
        return (float((Am-Bm)*(Am-Bm).T)/len(Am[0,:]))**0.5

def test(xn,yn,fn_values,domain):
    #print fn_values
    #test the X axis first
    #generate arrays of derivatives of points
    #from arrays of points of derivatives
    #print fn_values
    k=2
    D=map(lambda x: x[0],domain[0])
    print map(lambda x: (x,domain[0][4][0]),D)
    for i in range(3):
        j=0
        while j<=k:
            if  i <= xn and j < yn:
                fnox=fn_values[i][j][4]
                fnoxdx=fn_values[i+1][j][4]
                fnoy=map(lambda x:x[4],fn_values[i][j])
                fnoydy=map(lambda x:x[4],fn_values[i][j+1])
                fnox_aprox=integrate(fnoxdx,X=D,f_0=fnox[0])
                fnoy_aprox=integrate(fnoydy,X=D,f_0=fnoy[0])
                #print D
                print i,j, "function"
                print "want",fnoy
                print "got",fnoy_aprox
                #print fnoxdx
                #print fnox_Ix
                ex=Error(fnox_aprox,fnox)
                ey=Error(fnoy_aprox,fnoy)
                #print "error, fox/Ifoxdx, foy/Ifoydy",ex,ey
                #print "error, fox/Ifoxdx", ex
                print "error", ey
            j+=1
        k-=1
        
class McFall:
    def __init__(self, prob,xn=-1,yn=-1,interpolation_type = 'euclidean'):
        '''
        Create the explicit McFall representation and perform preprocessing to generate the components:\n
        Dirichlet boundary satisfying function  and derivatives: Ad_values_fns\n
        The extended Dirichlet boundary  and derivatives: extDirichlet\n
        The Dirichlet length factor function  and derivatives: LF_d\n
        Neumann boundary satisfying function and derivatives g(x): g_values_fns\n
        The Neumann length factor function  and derivatives: LF_m\n
        The x normals satisfying function: Normal_ox \n
        The y normals satisfying function: Normal_oy \n        
        '''
        print "preprocessing McFall-explicit components"
        Enum = ["Dirichlet only","Neumann only",
                          "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""] 
        if xn==-1 and yn==-1:
            xn,yn=prob.orderX,prob.orderY
        self.prob =prob
        self.Xgrid = prob.Xgrid
        self.Ygrid = self.prob.Ygrid

        o=0
        if self.prob.boundaryTypeId>0:
            o=1
        #take the boundary included domain, subset boundary points for interpolation
        #apply interpolation to all points not in boundary subset
        #Ad = f(x), values only
        print "Dirichlet components:"
        print "calculating boundary satisfying function Ad"
        approx = approxFunction(self.prob.Xgrid,self.prob.Ygrid,self.prob.boundDomain, self.prob.d_Ohm_D,\
                                self.prob.DC, 0, interpolation_type = interpolation_type,xn_max=xn+o,yn_max=yn+o)
        #self.Ad_values_fns = approx.calculateDerivatives([],xn_max=xn,yn_max=yn)
        self.Ad_values_fns = approx.fns

        print "extend dirichlet boundary"
        self.extDirichlet= self.extendBoundaries("Neumann","Dirichlet",self.prob.d_Ohm,theta=self.prob.theta,\
                                                 affineX = self.prob.Xaffine,affineY =  self.prob.Yaffine)
        #length factor functions   
        print "calculating length factor function Ad"
        #print "loading length factor function",self.prob.boundaryTypeId,Enum[self.prob.boundaryTypeId]
        #load LF_d, determine if a preloaded function exists
        self.LF_d = self.loadLengthFactor(Enum[self.prob.boundaryTypeId],'Dirichlet')
        if len(self.LF_d)==0:
            self.LF_d = self.lengthFactorFns(xn+o,yn+o,self.prob.F_list, self.extDirichlet,self.prob.theta, self.prob.boundDomain)

        self.mixed=False
        if len(self.prob.d_Ohm_N)>0:
            self.mixed=True
            print "Neumann components:"
            self.extNeumann = self.extendBoundaries("Dirichlet","Neumann",self.prob.d_Ohm,theta=self.prob.theta)
            #load LF_m, determine if a preloaded function exists
            print "Calculating Neumann length factor function L_M"
            self.LF_m = self.loadLengthFactor(Enum[self.prob.boundaryTypeId],'Neumann')
            if len(self.LF_m)==0:
                self.LF_m = self.lengthFactorFns(xn+o,yn+o,self.prob.F_list,self.extNeumann,self.prob.theta, self.prob.boundDomain)
                #print self.LF_m 
            #print extNeumann
            self.extNormals = determineNormals(self.extNeumann)
            print "calculating boundary satisfying function Am"
            approx = approxFunction(self.prob.Xgrid,self.prob.Ygrid,self.prob.boundDomain, self.prob.d_Ohm_N, \
                                    self.prob.NC, 0, interpolation_type = interpolation_type,xn_max=xn+o,yn_max=yn+o)
            #self.g_values_fns = approx.calculateDerivatives([],xn_max=xn,yn_max=yn)
            self.g_values_fns= approx.fns
##            print "Normal nx(x,y)"
            bound_weight=[]
            c=4.0
            d_ohm=range(len(self.prob.d_Ohm))
            print "calculating normal scalar function nx(x,y)"
            #bound_weight=map(lambda x: 1-0.2*math.exp(-(x-len(d_ohm)/2.0+1)**2/(2.0*c**2)),d_ohm)
            approx = approxFunction(self.prob.Xgrid,self.prob.Ygrid,self.prob.boundDomain,\
                                         map(lambda x:x[0],self.prob.d_Ohm), map(lambda x:x[0],self.prob.normal),\
                                         0,interpolation_type = interpolation_type,xn_max=xn+o,yn_max=yn+o,bound_weight=bound_weight)
            #self.Normal_ox = approx.calculateDerivatives([],xn_max=xn,yn_max=yn)
            self.Normal_ox= approx.fns
            print "calculating normal scalar function ny(x,y)"
            approx =approxFunction(self.prob.Xgrid,self.prob.Ygrid,self.prob.boundDomain,\
                                        map(lambda x:x[0],self.prob.d_Ohm), map(lambda x:x[1],self.prob.normal),\
                                        0,interpolation_type = interpolation_type,xn_max=xn+o,yn_max=yn+o,bound_weight=bound_weight)
            self.Normal_oy= approx.fns

    
    def LengthFactor(self,xn,yn,r,X,Y):
        '''
        The partial derivatives of the length factor function: xn, yn
        The radius, should be 1 for a unit circle...
        A matrix of derivatives of the input with respect to x and y
        '''
        lf=0
        if xn==0 and yn==0:
            lf=r - X[0][0]**2 - Y[0][0]**2
        elif xn==0 and yn==1:
            lf=-2*X[0][0]*X[0][1] - 2*Y[0][0]*Y[0][1]
        elif xn==0 and yn==2 :
            lf=-2*X[0][0]*X[0][2] - 2*Y[0][0]*Y[0][2] - 2*X[0][1]**2 - 2*Y[0][1]**2
        elif xn==0 and yn==3 :
            lf=-2*X[0][0]*X[0][3] - 2*Y[0][0]*Y[0][3] - 6*X[0][1]*X[0][2] - 6*Y[0][1]*Y[0][2]
        elif xn==0 and yn==4:
            lf=-2*X[0][0]*X[0][4] - 2*Y[0][0]*Y[0][4] - 8*X[0][1]*X[0][3] - 8*Y[0][1]*Y[0][3] - 6*X[0][2]**2 - 6*Y[0][2]**2
        elif xn==1 and yn==0 :
            lf=-2*X[0][0]*X[1][0] - 2*Y[0][0]*Y[1][0]
        elif xn==1 and yn==1 :
            lf=-2*X[0][0]*X[1][1] - 2*Y[0][0]*Y[1][1] - 2*X[1][0]*X[0][1] - 2*Y[1][0]*Y[0][1]
        elif xn==1 and yn==2 :
            lf=-2*X[0][0]*X[1][2] - 2*Y[0][0]*Y[1][2] - 2*X[1][0]*X[0][2] - 4*X[0][1]*X[1][1] - 2*Y[1][0]*Y[0][2] - 4*Y[0][1]*Y[1][1]
        elif xn==1 and yn==3:
            lf=-2*X[0][0]*X[1][3] - 2*Y[0][0]*Y[1][3] - 2*X[1][0]*X[0][3] - 6*X[0][1]*X[1][2] - 2*Y[1][0]*Y[0][3] - 6*Y[0][1]*Y[1][2] - 6*X[1][1]*X[0][2] - 6*Y[1][1]*Y[0][2]
        elif xn==2 and yn==0 :
            lf=-2*X[0][0]*X[2][0] - 2*Y[0][0]*Y[2][0] - 2*X[1][0]**2 - 2*Y[1][0]**2
        elif xn==2 and yn==1 :
            lf=-2*X[0][0]*X[2][1] - 2*Y[0][0]*Y[2][1] - 4*X[1][0]*X[1][1] - 2*X[0][1]*X[2][0] - 4*Y[1][0]*Y[1][1] - 2*Y[0][1]*Y[2][0]
        elif xn==2 and yn==2:
            lf=-2*X[0][0]*X[2][2] - 2*Y[0][0]*Y[2][2] - 4*X[1][0]*X[1][2] - 4*X[0][1]*X[2][1] - 4*Y[1][0]*Y[1][2] - 4*Y[0][1]*Y[2][1] - 2*X[2][0]*X[0][2] - 4*X[1][1]**2 - 2*Y[2][0]*Y[0][2] - 4*Y[1][1]**2
        elif xn==3 and yn==0 :
            lf=-2*X[0][0]*X[3][0] - 2*Y[0][0]*Y[3][0] - 6*X[1][0]*X[2][0] - 6*Y[1][0]*Y[2][0]
        elif xn==3 and yn==1:
            lf=-2*X[0][0]*X[3][1] - 2*Y[0][0]*Y[3][1] - 6*X[1][0]*X[2][1] - 2*X[0][1]*X[3][0] - 6*Y[1][0]*Y[2][1] - 2*Y[0][1]*Y[3][0] - 6*X[2][0]*X[1][1] - 6*Y[2][0]*Y[1][1]
        elif xn==4 and yn==0:
            lf=-2*X[0][0]*X[4][0] - 2*Y[0][0]*Y[4][0] - 8*X[1][0]*X[3][0] - 8*Y[1][0]*Y[3][0] - 6*X[2][0]**2 - 6*Y[2][0]**2
        return float(lf)
    
    def lengthFactorDerivatives(self, xn,yn, point):
        '''
        The partial derivatives of the length factor function array, xn and yn\n
        A set of transformed domain points and derivatives within the unit circle - point\n
        Returns the length factor values and derivatives over the domain\n
        '''
        Lf = []
        
        X=[map(lambda x: x[0],point[0]),map(lambda x: x[0],point[1]),map(lambda x: x[0],point[2]),map(lambda x: x[0],point[3]),map(lambda x: x[0],point[4])]
        Y=[map(lambda x: x[1],point[0]),map(lambda x: x[1],point[1]),map(lambda x: x[1],point[2]),map(lambda x: x[1],point[3]),map(lambda x: x[1],point[4])]
        k=4
        for i in range(5):
            j=0
            row=[]
            while j <= k:

                if i<=xn and j<=yn:
                    #print "inside"
                    r=1
                    #print X
                    #print Y
                    #print i,j,func["length_factors_"+str(i)+"_"+str(j)]
                    #entry = eval(func["length_factors_"+str(i)+"_"+str(j)])
                    entry = self.LengthFactor(i,j,r,X,Y)
                    #print entry,entry1
                    row += [entry]
                else:
                    row+=[0]
                j+=1
            Lf += [row]    
            k-=1
       # print "finish"
        return Lf

    def lengthFactorFns(self,xn,yn, F_list, boundary,theta, domain):
        '''
        For all derivatives, up to xn,yn, over the domain, using the boundary points, create a Thin
        Plate Spline transformation mapping between the domain and the boundary points, apply the
        euclidean distance from the radius of a unit circle.
        '''
        print "calculate length factors"
        target = self.targetCircle(theta = theta)
       # print target
        transform = ThinPlateSpline(boundary,target)
##        fns = []
##        k=3
##        print "start0"
##        for i in range(4):
##            j=0
##            row=[]
##            while j <= k:
##                print i,j,k
##                row += [transform.transformDomain(i,j,domain)]
##                j+=1
##            fns += [row]
##            k-=1
##        print "finish0"
##        Xarray=[]
##        Yarray=[]
##        c=0
        Lf = []
        E_fns = []
        C_fns = []
        A_fns=[]
        time1=time()
        for i in range(len(domain)):
##            Xarrayrow=[]
##            Yarrayrow=[]
            row_lf=[]
            row_a=[]
            row_c=[]
            row_e=[]
            for j in range(len(domain[0])):
                print i,j
                E=transform.preProcessEuclidean(xn,yn,domain[i][j])
                #print "euclidean\n", E
                C=transform.preProcessTransform(xn,yn,domain[i][j],E)
                #print "C\n", C
                Apoint=transform.transformDerivatives(xn,yn,domain[i][j],C)
               # print "Apoint\n", Apoint
                #select an arbirtary control point to be a constant E[0],C[0]
                row_e += [E[3]]
                row_c += [C[3]]
                q=0
               # print Apoint
                row_a += [[map(lambda x: x[q],Apoint[0]),map(lambda x: x[q],Apoint[1]),map(lambda x: x[q],Apoint[2]),map(lambda x: x[q],Apoint[3])]]
                LF = self.lengthFactorDerivatives(xn,yn, Apoint)
                #print LF
                row_lf += [LF]

            Lf+=[row_lf]
            E_fns+=[row_e]
            C_fns+=[row_c]
            A_fns+=[row_a]

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
                            domain_row+=[Lf[i0][j0][i][j]]
                        Domain+=[domain_row]
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
                    row+=[Domain]
                    j+=1
                Derivatives+=[row]
                k-=1
        time2=time()
        print "time ", time2-time1
        return Lf
    def loadLengthFactor(self,boundaryCombination, boundSelect):
        '''
        load pre-generated length factor functions.

        '''
        new_dir = os.getcwd() + '\\preprocessed_info'
        #os.chdir(new_dir)
        if self.prob.numpoints==9:
            if boundaryCombination == 'Dirichlet only':
                LF_file = open(new_dir + "\\LF_single_rect.txt","r")
                lf_str = LF_file.readlines()
                LengthFactor=eval(lf_str[0])
            elif boundaryCombination == 'Arbitrary Mixed Dirichlet/Neumann':
                if boundSelect == "Dirichlet":
                    LF_file = open(new_dir + "\\LF_arb_dirichlet_rect.txt","r")
                elif boundSelect == "Neumann":
                    LF_file = open(new_dir + "\\LF_arb_neumann_rect.txt","r")
                lf_str = LF_file.readlines()
                LengthFactor=eval(lf_str[0])      
            elif boundaryCombination == 'Random Mixed Dirichlet/Neumann':
                if boundSelect == "Dirichlet":
                    LengthFactor=[]
                elif boundSelect == "Neumann":
                    LengthFactor=[]
            elif  boundaryCombination == 'Neumann only':
                if boundSelect == "Dirichlet":
                    LF_file = open(new_dir + "\\LF_encompassing_rect.txt","r")
                elif boundSelect == "Neumann":
                    LF_file = open(new_dir + "\\LF_single_rect.txt","r")
                lf_str = LF_file.readlines()
                LengthFactor=eval(lf_str[0])        
            else:
                LengthFactor=[]
        else:
            LengthFactor=[]    
        return LengthFactor
    def viewComponent(self, component,xn,yn):
        "view the component with the given partial derivatives xn, yn"
        if component=="ad":
            self.view(self.Ad_values_fns,xn,yn,text = "Ad")
        elif component=="ld":
            self.view(self.LF_d,xn,yn,text = "Length Factor Dirichlet")
        if len(self.prob.d_Ohm_N)>0:
            if component=="g":
                self.view(self.g_values_fns,xn,yn,text = "g")
            elif component=="lm":
                self.view(self.LF_m,xn,yn,text = "Length Factor Neumann")
            elif component=="nx":
                self.view(self.Normal_ox,xn,yn,text = "nx")
            elif component=="ny":
                self.view(self.Normal_oy,xn,yn,text ="ny" )
            elif component=="bound":
                self.view_bound()

    def view(self, component, xn,yn,text ='Source Function' ):
        #print component
        Z=[]
        for i in range(len(component)):
            row=[]
            for j in range(len(component[0])):
                
                row+=[component[i][j][xn][yn]]
            Z+=[row]
    
        
        fig = plt.figure()
        fig.suptitle(text)
        ax = fig.gca(projection='3d')

        X,Y = np.meshgrid(self.Xgrid,self.Ygrid)
        Z = np.array(Z)
        #print Z
        surf = ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap=cm.jet,
                               linewidth=0, antialiased=False)
        function_height = Z.max()-Z.min()
        ax.set_zlim3d(Z.min()-0.2*function_height,Z.max()+0.2*function_height)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ##
        fig.colorbar(surf, shrink=0.5,aspect=5)
        plt.show()

    def view_bound(self):
        
        mpl.rcParams['legend.fontsize'] = 10
        D=map(lambda x:x[0],self.prob.d_Ohm)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.array(map(lambda x:x[0],self.extDirichlet))
        y = np.array(map(lambda x:x[1],self.extDirichlet))
        z = np.array(len(x)*[0])
        ax.plot(x, y, z, label='New Dirichlet Boundary')
        if len(self.prob.d_Ohm_N)>0:
            x = np.array(map(lambda x:x[0],self.extNeumann))
            y = np.array(map(lambda x:x[1],self.extNeumann))
            z = np.array(len(x)*[0])
            ax.plot(x, y, z, label='New Neumann Boundary')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
    
    def localPoint(self, point, xaffine,yaffine):

        new_point= [0,0]
        #for i in range(len(domain)):
            #for j in range(len(domain)):
        new_point[0] = xaffine[0,0]*point[0] + xaffine[1,0]
        new_point[1] = yaffine[0,0]*point[1] + yaffine[1,0]
        return new_point

    #give this targetmapping object later
    def targetCircle(self, num_points=-1, theta=[]):
        if theta == []:
            theta = numpy.linspace(0,2*pi,num_points)
        circle = []
        for t in theta:
            circle += [[cos(t), -sin(t)]]
        return circle
    #requires a list boundarypoints of the form [[[x,y],"boundary1"],...]
    #extend boundary1
    def extendBoundaries(self,boundary1, boundary2, boundarypoints, num_points=-1, theta=[],affineX=matrix([[1],[0]]),affineY=matrix([[1],[0]])):
        '''if boundary1 and boundary2 are the subsections of a domain, then this method extends boundary1 about boundary 2
        so intersections occur outside the boundary.'''
        print "extend " + boundary1 +" boundary about the " + boundary2 +" boundary" 
        ext=1.05
        referencePoints = self.targetCircle(theta = theta)
        modCircle = []
        
        for i,t in enumerate(boundarypoints):
            if boundarypoints[i][1] == boundary1:
                #print boundarypoints[i][0][0],boundarypoints[i][0][1]
                modCircle +=[[ext*referencePoints[i][0],ext*referencePoints[i][1]]]
            else:
                modCircle +=[referencePoints[i]]
        boundarypoints = map(lambda x: x[0], boundarypoints)
        #local_boundarypoints = map(lambda x: self.localPoint(x,affineX,affineY), boundarypoints)
        
        #print zip(local_boundarypoints,boundarypoints), 
        mod_boundary = []
        #print referencePoints
        #print boundarypoints
        transform = ThinPlateSpline(referencePoints,boundarypoints)
        #transform.transform(0,0,[0.5,0.5])
        for i in range(len(modCircle)):
            E = transform.preProcessEuclidean(0,0,modCircle[i])
            C = transform.preProcessTransform(0,0,modCircle[i],E)
            mod_boundary += [transform.transform(0,0,modCircle[i],C)];
        return mod_boundary
