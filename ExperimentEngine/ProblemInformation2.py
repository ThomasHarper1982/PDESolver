from sympy import *
import numpy
from numpy import matrix
import math
import random
import copy
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

class ProblemInformationList:
    def __init__(self):
        self.prob_list = []
        self.max_num = 0
        
    def insert(self, name):
        self.prob_list += ProblemInformation()
        self.max_num +=1

    def delete(self, name):
        self.max_num -=1

class ProblemInformation:
    def __init__(self, name=''):
        self.name = name
    
##    def dimLists(self, dim, n):
##        Lists = []
##        if dim < 1:
##            return [0]
##        else:
##            for j in range(n):
##                Lists+=[self.dimLists(dim-1,n)]
##        return Lists

    #set the Domain, the function of the Domain#
    def setTargetMapping(self, Domain, mapping,dim):
        self.Domain = Domain
        self.mapping = mapping
        self.dim = dim

    #specify points and grid size dimension
    def setDomain(self, Domain=[],point1=[], point2=[],numpoints=0):
        if Domain != []:
            self.Domain = Domain
        elif numpoints>0:
            Domain=[]
            boundDomain=[]
            if self.dim==1:
                self.point1= point1
                self.point2 = point2
            elif self.dim==2:
                X0 = point1[0]
                X1 = point2[0]
                Y0 = point1[1]
                Y1 = point2[1]
                self.Xaffine    = matrix([[X0, 1],[X1, 1]]).I*matrix([-1,1]).T
                self.Yaffine    = matrix([[Y0, 1],[Y1, 1]]).I*matrix([-1,1]).T
                self.Xgrid = numpy.linspace(X0,X1,numpoints)
                self.Ygrid = numpy.linspace(Y0,Y1,numpoints)
                #print Xaffine, Yaffine
               # print len(Xgrid)
                for i in range(1,len(self.Ygrid)-1):
                    Domain+=[map(lambda x: [x,self.Ygrid[i]], self.Xgrid[1:len(self.Xgrid)-1])]
                for i in range(0,len(self.Ygrid)):
                    boundDomain+=[map(lambda x: [x,self.Ygrid[i]], self.Xgrid[0:len(self.Xgrid)])]

            print boundDomain
            self.clippedDomain = copy.deepcopy(Domain)
            self.Domain= flatten(Domain, levels=1)
            self.boundDomain = boundDomain
            self.numBoundDomain = numpoints**2
            self.point1=point1
            self.point2=point2
            self.hx = self.Xgrid[1] - self.Xgrid[0]
            self.hy = self.Ygrid[1] - self.Ygrid[0]
    #set del Omega for rectangular orthogonal Domain, num_points - 1 divides 8
    def setBoundaryPoints(self,num_points):
##        self.num_points = num_points
##        r=1;
##        pi=math.pi
##        sin=math.sin
##        theta = numpy.linspace(0,2*pi,num_points)
##        num_points-=1
##        theta = theta[0:num_points]
##        self.theta= theta
##        self.d_Ohm= []
##        section = floor(num_points/8);
##        offset = 0;
##        #print theta
##        #consider whether to include the first point twice
##        for i in range(num_points):
##            #print i
##            E=[]
##            if 0+offset < theta[i] and theta[i] <= pi/4+offset:
##                E = [1,sin(2*theta[i])]
##            elif pi/4+offset < theta[i] and theta[i] <= pi/2+offset or pi/2+offset < theta[i] and theta[i] <= 3*pi/4+offset:
##                E = [sin(2*theta[i]),1]
##            elif 3*pi/4+offset < theta[i] and theta[i] <= pi+offset or pi+offset < theta[i] and theta[i] <= 5*pi/4+offset:
##                E = [-1,-sin(2*theta[i])]
##            elif 5*pi/4+offset < theta[i] and theta[i] <=3*pi/2+offset or 3*pi/2+offset< theta[i] and theta[i] <= 7*pi/4+offset:
##                E = [-sin(2*theta[i]),-1]
##            elif 7*pi/4+offset < theta[i] and theta[i] <= 2*pi or  0 <= theta[i] and theta[i] <= offset:
##                E = [1,sin(2*theta[i])]
##            E[0] = (E[0] - self.Xaffine[1,0])/self.Xaffine[0,0]
##            E[1] = (E[1] - self.Yaffine[1,0])/self.Yaffine[0,0]
##            self.d_Ohm+=[E]
##            self.num_points = num_points
        r=1
        pi=math.pi
        sin=math.sin
        num_points=4*int(sqrt(self.numBoundDomain))-4
        theta = numpy.linspace(0,2*pi,num_points+1)
        #num_points-=1
        theta = theta[0:num_points]
        self.theta= theta
        self.d_Ohm= []
        section = floor(num_points/8);
        offset = 0;
        #print theta
        #consider whether to include the first point twice
        end = len(self.boundDomain)-1
        start =0
        i=int(len(self.boundDomain[0])/2.0)
        j=end
        #print 'start' , i,j, theta[0],num_points, sqrt(len(prob.boundDomain))
        #print self.boundDomain
        for n in range(num_points):
            #print n,i,j, theta[n],prob.boundDomain[i][j],self.point1,self.point2
            self.d_Ohm+=[self.boundDomain[i][j]]        
            if self.boundDomain[i][j][0] == self.point2[0] and self.boundDomain[i][j][1] != self.point1[1]:
                [i,j]=[i-1,j]
            elif self.boundDomain[i][j][1] == self.point1[1] and self.boundDomain[i][j][0] != self.point1[0]:
                [i,j]=[i,j-1]
            elif self.boundDomain[i][j][0] == self.point1[0] and self.boundDomain[i][j][1] != self.point2[1]:
                [i,j]=[i+1,j]
            elif self.boundDomain[i][j][1] == self.point2[1] and self.boundDomain[i][j][0] != self.point2[0]:
                [i,j]=[i,j+1]

    #requires the domains to be set
    def setFunctions(self):
        #store the values of the function and derivs
        self.F = []
        for i in range(len(self.Domain)):
            self.F += [self.dimLists(self.dim,4)]
                #determine DE_p
        self.F_p = []
        for i in range(len(self.Domain)):
            self.F_p += [self.dimLists(self.dim,4)]  
    
    #set the Dirichlet boundary domain Omega
    def setDirichlet(self, d_Ohm_D,DC=[]):
        self.d_Ohm_D = d_Ohm_D
        self.DC = DC
        
    #set the Neuman boundary domain
    def setNeumann(self, d_Ohm_N, NC=[], normal=[]):
        self.d_Ohm_N = d_Ohm_N
        self.NC = NC
        self.normal =normal
        
    #set the dimension
    def setDim(self, dim):
        self.dim = dim

    #set basis function, 'sigmoid', 'radial', 'sine'
    def setBasis(self, b):
        self.b = b

    #representation, implicit, explicit
    def setRepresention(self, r):
        self.r = r

    def matchBracket(self, s, i, bracket):
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

    #assumes we have set the differential equation
    #derive the source function from differential equation and solution information
    def deriveSourceFunction(self, sln):
        #differentiate the sln based on the F_list
        x=Symbol('x')
        y=Symbol('y')
        #print eval(sln)
        F=[]
        if self.dim==2:
            F =  self.dimLists(self.dim,4)
            F[0][0] = eval(sln)
        elif self.dom==1:
            F =[0,0,0,0]
            F[0] = eval(sln)
        #DE_aux = self.DE.replace("F", "Sln")
       # print F
        for i,n in enumerate(self.F_list):
            if self.dim==2:
                #print n[0],n[1], eval(sln)
                F[n[0]][n[1]] = diff(diff(eval(sln),x,n[0]),y,n[1])
                #F[n[0]][n[1]] = diff(eval(sln),x,n[0],y,n[1])
                #print F[n[0]][n[1]]
            elif self.dom==1:
                F[n[0]] = diff(eval(sln),x,n[0])
        self.sourceFunction = str(simplify(eval(self.DE)))
        sourceFunction=self.sourceFunction
        self.SF = []
        sourceFunction=sourceFunction.replace("*cos","*math.cos")
        sourceFunction=sourceFunction.replace("*sin","*math.sin")
        sourceFunction=sourceFunction.replace("*exp","*math.exp")
        sourceFunction=sourceFunction.replace("*log","*math.log")
        sourceFunction=sourceFunction.replace("pi",str(math.pi))
        for i in range(len(self.Domain)):
            x = self.Domain[i][0]
            y = self.Domain[i][1]
            self.SF += [eval(sourceFunction)]
        print self.sourceFunction
        #print self.SF
        self.SF2 = [] 
        for i in range(len(self.boundDomain)):
            row=[]
            for j in range(len(self.boundDomain[0])):
                x = self.boundDomain[i][j][0]
                y = self.boundDomain[i][j][1]
                row+=[eval(sourceFunction)]
            self.SF2 += [row] 
                
    def setBoundarySegments(self):
        #circle_points = length(self.theta);
        self.d_Ohm_N = [];
        self.d_Ohm_D = [];
        j=0
        #print len(self.theta)
        #print self.Angles
        for i in range(len(self.theta)):
            angle = self.theta[i];
            
            if j < len(self.Angles)-1:
            
               # a = self.Angles[j][1]
                b = self.Angles[j+1][1]
            else:
                #a=self.Angles[j][1]
                b=2*pi
            if angle > b:
                j+=1

            #print i, angle, a , b, self.Angles[j][0]
            #if a <= angle and angle < b:
            if self.Angles[j][0] == "Dirichlet":
                self.d_Ohm_D += [self.d_Ohm[i]]
                self.d_Ohm[i] = [self.d_Ohm[i],"Dirichlet"]   
            elif self.Angles[j][0] == "Neumann":
               # print self.d_Ohm[i]
                self.d_Ohm_N += [self.d_Ohm[i]]
                self.d_Ohm[i] = [self.d_Ohm[i],"Neumann"]

           # else:
             #   j+=1      
        print "Dirichlet",self.d_Ohm_D
        print "Neumann",self.d_Ohm_N
        print "combined", self.d_Ohm

    def setDirichletNeumannAngles(self, boundaryTypeId,numAngles=2,nstate = round(random.random())):
        self.Angles =[];
        if boundaryTypeId == 0:
            self.Angles = [("Dirichlet",0)]
        elif boundaryTypeId == 1:
            self.Angles = [("Neumann",0)]     
        elif boundaryTypeId == 2:
            self.Angles = [("Dirichlet",0),("Neumann",0.25*pi), ("Dirichlet",0.75*pi)]
        elif boundaryTypeId == 3:
           #numAngles = 1+num_angles*round(random.random())
           angle =0;
           slices = 2*math.pi/numAngles;
           #split evenly into d+n segments of either kind 
           d=1;
           n=1;
           for i in range(numAngles):
              if nstate:
                  self.Angles += [("Neumann",angle)]
                  nstate=0
              else:
                  self.Angles += [("Dirichlet",angle)]
                  nstate=1
              angle = angle + slices
        #print self.Angles

    def setNormals(self):
        self.normal = []
        #change to let the normals correspond to d_Ohm rather than d_Ohm_N
        for i,n in enumerate(map(lambda x: x[0],self.d_Ohm)):
            #print i,n
##            if n[0]==self.point1[0] or n[0] ==self.point2[1]:
##                self.normal += [[1,0]]
##            elif n[1]==self.point1[1] or n[1] ==self.point2[1]:
##                self.normal += [[0,1]]
       # print self.normal
            if n[0]==self.point1[0]:
                self.normal += [[1,0]]
            elif n[0]==self.point2[0]:
                self.normal += [[-1,0]]
            elif n[1]==self.point1[1]:
                self.normal += [[0,1]]
            elif n[1]==self.point2[1]:
                self.normal += [[0,-1]]    
            
    def setBoundaryValues(self, sln):
        #dirichlet
        self.DC=[]
        sln_aux=self.markupString(sln)
        #print sln
        for d in self.d_Ohm_D:
            x = d[0]
            y = d[1]
            self.DC+= [eval(sln_aux)]
            #print d, sln_aux, eval(sln_aux)
        #print "Dirichlet", self.DC
        #neumann
        #obtain normalsself.
        self.setNormals()
        self.NC=[]
        #print sln
        x=Symbol('x')
        y=Symbol('y')
        sln_x = self.markupString(str(diff(eval(sln),x)))
        sln_y = self.markupString(str(diff(eval(sln),y)))
       # print len(self.d_Ohm_N) , len(self.normal)
        for i,n in enumerate(self.d_Ohm_N):
            #print n
            x = n[0]
            y = n[1]
            #print eval(sln_x), eval(sln_y)
           # print i
            self.NC+= [eval(sln_x)*self.normal[i][0]+eval(sln_y)*self.normal[i][1]]
            #print n, sln, eval(sln)
       # print "Neumann", self.NC

            
    def markupString(self, exp):
        exp=exp.replace("cos","math.cos")
        exp=exp.replace("sin","math.sin")
        exp=exp.replace("exp","math.exp")
        exp=exp.replace("log","math.log")
        exp=exp.replace("pi","math.pi")
        return exp

    #creates data structures F, F_list, F_P for use in the energy function and derives derivative DE_p
    def setDifferentialEquation(self, DE,  sourceFunction='0', dim =1, sln="", Domain=[],boundmode =0, ax=1, bx=0,ay=1,by=0):

       # self.dim =dim
        self.sourceFunction='0'
        print DE
        self.DE = DE

        #F_indices store the indices
        self.F_list = []
        #search for F, then 2 indices
        i = 0
        while i != -1:
            i = (self.DE).find('F',i)
            j=0
            tup = [0]*self.dim
            if i != -1:
                while j < self.dim and i < len(DE):
                    if self.DE[i].isdigit():
                        tup[j] =int(self.DE[i])
                        j+=1
                    i+=1
                self.F_list += [tup]
        self.orderX = max(map(lambda x: x[0], self.F_list))
        self.orderY = max(map(lambda x: x[1], self.F_list))
        #extract source function last
        #self.sourceFunction = sourceFunction
       # print self.F_list
        self.F_p_list = []
        x=Symbol('x')
        y=Symbol('y')
        param = Symbol('param')
        Function_str = []
        DE_Aux = self.DE[:]
        DE_Aux =DE_Aux.replace("math.", "")
        F =  self.dimLists(self.dim,4)
        for i,n in enumerate(self.F_list):
            fn_str = "F"
            for j,v in enumerate(n):
                fn_str += '[' + str(v) + ']'
            
            Function_str += [fn_str+"(param)"]
            DE_Aux =DE_Aux.replace(fn_str, Function_str[i])
            if self.dim ==2:
                #print Function_str[i]
                F[n[0]][n[1]] = Function(Function_str[i])
            elif self.dim ==1:
                F[n[0]] = Function(Function_str[i])
       # print DE_Aux
        D_p_string = str(diff(eval(DE_Aux),'param'))
       # print D_p_string, type(D_p_string)
        D_p_string=D_p_string.replace("(param)","")
       # print D_p_string, type(D_p_string)
        #remove (param)
        #find D index, identify l.h bracket and matching r.h bracket
        for j,n in enumerate(self.F_list):
            i=0
            #while i != -1:
            i = D_p_string.find("D", 0)
            start_i = i
            i+=2
           # print "start", start_i
            end_i = self.matchBracket(D_p_string, i, '(')
           # print "end", end_i
            fn_str = "F_p"
            for j,v in enumerate(n):
                fn_str += '[' + str(v) + ']'
            D_p_string= D_p_string[0:start_i] + fn_str+ D_p_string[end_i+1:len(D_p_string)]
           # print D_p_string
        D_p_string=self.markupString(D_p_string)
        self.setFunctions()
       # print D_p_string
        self.DE_p = D_p_string
        
        #identify indices
##            tup = [0]*self.dim
##            while j < self.dim and i < end_i:
##                if self.DE[i].isdigit():
##                    tup[j] =int(self.DE[i])
##                    j+=1
##                i+=1
##prob = ProblemInformation()
##prob.setDim(2)
##prob.setDomain(map(lambda x,y: [x,y], numpy.linspace(0,1,10), numpy.linspace(0,0,10)))
##prob.setRepresention('implicit')
##prob.setDifferentialEquation("x**2*math.sin(F[0][0])-math.cos(x)*F[1][0]**2")

##prob = ProblemInformation()
##prob.setDim(2)
#####prob.setDomain(numpy.linspace(0,1,10))
##prob.setDomain(point1 = [0,0], point2=[1,1],numpoints=9)
#####print prob.boundDomain
##n=3
##prob.setBoundaryPoints(8*n+1)
##prob.setRepresention('implicit')
#####prob.setDifferentialEquation("x**2*math.sin(F[0][0])-math.cos(x)*F[1][0]**2")
##prob.setDifferentialEquation("F[2][0]+F[0][2]")
##prob.setDirichletNeumannAngles(2,numAngles=3,nstate=1)
##prob.setBoundarySegments()
##prob.setBoundaryValues("y**2*sin(pi*x)")
###prob.deriveSourceFunction("y**0*(x+y**3)")
##prob.deriveSourceFunction("y**2*sin(pi*x)")
###print prob.boundDomain
##X= prob.NC

##diff= map(lambda x,y: x-y, X,X[1:len(X)]+[X[0]])
##print diff
##print X
##print "std " , sqrt(sum(map(lambda x: (sum(diff)/len(diff)-x)**2, diff)))/(len(diff))

