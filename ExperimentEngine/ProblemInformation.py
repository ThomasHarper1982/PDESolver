# Represents a differential equation and boundary value problem

from sympy import *
import numpy
from numpy import matrix
import math
import random
import copy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

#holds a list ProblemInformation objectsm this was added after ProblelmInformation
#was created when it is was decided that multiple differential equations with
#consistent solutions may want to be treated (to add constraints)
#for example, F[0][0]=s(x),F[0][1]=s_y(x),F[1][0]=s_x(x) maybe imposed to assess
#how well the points are interpolated AND derivatives are approximated
class ProblemInformationList:
    def __init__(self, name=""):
        self.prob_list = []
        self.unique = 0
        self.name=name
        
    def insert(self):
        prob = ProblemInformation(name='term '+str(self.unique))
        self.prob_list += [prob]
        self.unique +=1
        return prob

    def delete(self, i):
##        self.max_num -=1
##        for i, prob in enumerate(self.prob_list):
##            if prob.name == name:
##                self.prob_list=self.prob_list[0:i-1] +self.prob_list[i+1:]
##                self.max_num -=1
##                break
        ###print self.prob_list
        self.prob_list=self.prob_list[0:i] +self.prob_list[i+1:]



# Produces and contains information for function approximation
# and differential equation solving, including the differential
# equation, the Dirichlet, Neumann conditions, the interior normals
# the boundary segments, the domain, the known solution.
class ProblemInformation:
    def __init__(self, name='',prob_type='de'):
        self.prob_type=prob_type
        self.sln=""
        self.name = name
        self.dim=2
        self.SF2=[]
    def dimLists(self, dim, n):
        Lists = []
        if dim < 1:
            return [0]
        else:
            for j in range(n):
                Lists+=[self.dimLists(dim-1,n)]
        return Lists

    def viewSourceFunction(self):
        if len(self.SF2)>0:
            fig = plt.figure()
            fig.suptitle('Source Function')
            ax = fig.gca(projection='3d')

            X,Y = np.meshgrid(self.Xgrid,self.Ygrid)
            SF2 = np.array(self.SF2)
            surf = ax.plot_surface(X,Y,SF2,rstride=1, cstride=1, cmap=cm.jet,
                                   linewidth=0, antialiased=False)
            function_height = SF2.max()-SF2.min()
            ax.set_zlim3d(SF2.min()-0.2*function_height,SF2.max()+0.2*function_height)
            ax.w_zaxis.set_major_locator(LinearLocator(10))
            ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
            ##
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(surf, shrink=0.5,aspect=5)
            plt.show()
    def viewSolution(self):
        if len(self.Sln2)>0:
            fig = plt.figure()
            fig.suptitle('Solution')
            ax = fig.gca(projection='3d')

            X,Y = np.meshgrid(self.Xgrid,self.Ygrid)
            Sln2 = np.array(self.Sln2)
            surf = ax.plot_surface(X,Y,Sln2,rstride=1, cstride=1, cmap=cm.jet,
                                   linewidth=0, antialiased=False)
            function_height = Sln2.max()-Sln2.min()
            ax.set_zlim3d(Sln2.min()-0.2*function_height,Sln2.max()+0.2*function_height)
            ax.w_zaxis.set_major_locator(LinearLocator(10))
            ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(surf, shrink=0.5,aspect=5)
            plt.show()
    def viewBoundary(self):
        #break the Dirichlet segments into discontiguous lists
        DirichletSegments = []
        DirichletSeg = []
        DirichletSegmentsValue=[]
        DirichletSegValue=[]
        NeumannSegments = []
        NeumannSeg = []
        NeumannSegmentsValue=[]
        NeumannSegValue=[]
        DirichletToggle = int(self.d_Ohm[0][1]=='Dirichlet')
        d_count=0
        n_count=0
       # print "Dirichlet ", self.DC
       # print "Neumann" ,self.NC
        for b in self.d_Ohm:
            if b[1] == 'Dirichlet':
               # print "add Dirichlet ", DirichletToggle
                if DirichletToggle == 1:
                    DirichletToggle -=1
                    if len(NeumannSeg)>0:
                        NeumannSegments+=[NeumannSeg]
                        NeumannSegmentsValue+=[NeumannSegValue]
                        NeumannSeg=[]
                        NeumannSegValue=[]
                DirichletSeg+=[b[0]]
                DirichletSegValue+=[self.DC[d_count]]
                #print b[0],b[1],self.DC[d_count]
                d_count+=1
            elif b[1] == 'Neumann':
                #print "add Neumann ",DirichletToggle
                if DirichletToggle == 0:
                    DirichletToggle +=1
                    if len(DirichletSeg)>0:
                        DirichletSegments+=[DirichletSeg]
                        DirichletSegmentsValue+=[DirichletSegValue]
                        DirichletSeg=[]
                        DirichletSegValue = []
                NeumannSeg+=[b[0]]
                NeumannSegValue+=[self.NC[n_count]]
               # print b[0],b[1],self.NC[n_count]
                n_count+=1
                
        if len(NeumannSeg)>0:
            NeumannSegments+=[NeumannSeg]
            NeumannSegmentsValue+=[NeumannSegValue]
            
        if len(DirichletSeg)>0:
            DirichletSegments+=[DirichletSeg]
            DirichletSegmentsValue+=[DirichletSegValue]


        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle('Dirichlet and Neumann Boundary Conditions')
        #---- First subplot
        ax = fig.add_subplot(2, 2, 1, projection='3d')
       # mpl.rcParams['legend.fontsize'] = 10
        #ax.axis('off')
        for i in range(len(DirichletSegmentsValue)):
            z = np.array(DirichletSegmentsValue[i])
            x = np.array(map(lambda x:float(x[0]),DirichletSegments[i]))
            y = np.array(map(lambda x:float(x[1]),DirichletSegments[i]))

            ax.plot(x, y, z, color="blue", label = "Dirichlet "+str(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #---- Second subplot
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        #mpl.rcParams['legend.fontsize'] = 10
        for i in range(len(NeumannSegmentsValue)):
            z = np.array(NeumannSegmentsValue[i])
            x = np.array(map(lambda x:float(x[0]),NeumannSegments[i]))
            y = np.array(map(lambda x:float(x[1]),NeumannSegments[i]))

            ax.plot(x, y, z, color="red", label = "Neumann "+str(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
##            ax.set_xbound((0, 1))
##            ax.set_ybound((0, 1))
            #ax.legend()
            #ax.axis('off')
        #ax.legend()
        #---- Third subplot
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        #mpl.rcParams['legend.fontsize'] = 10
        #print "Dirichlet segments"
        for i in range(len(DirichletSegmentsValue)):
            z = np.array([0]*len(DirichletSegments[i]))
            x = np.array(map(lambda x:float(x[0]),DirichletSegments[i]))
            y = np.array(map(lambda x:float(x[1]),DirichletSegments[i]))
          #  print zip(x,y)
            ax.set_xlim3d(self.point1[0], self.point2[0])
            ax.set_ylim3d(self.point1[1], self.point2[1])
            ax.plot(x, y, z, color="blue")#, label = "Dirichlet "+str(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        #print "Neumann segments"    
        #---- Second subplot
        #mpl.rcParams['legend.fontsize'] = 10
        for i in range(len(NeumannSegmentsValue)):
            z = np.array([0]*len(NeumannSegments[i]))
            x = np.array(map(lambda x:float(x[0]),NeumannSegments[i]))
            y = np.array(map(lambda x:float(x[1]),NeumannSegments[i]))
           # print zip(x,y)
            ax.plot(x, y, z, color="red")#, label = "Neumann "+str(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.legend()

        plt.show()


    #set the Domain, the function of the Domain#
    def setTargetMapping(self, Domain, mapping,dim):
        self.Domain = Domain
        self.mapping = mapping
        self.dim = dim
 
    #specify points and grid size dimension
    def setDomain(self, Domain=[],point1=[], point2=[],numpoints=0):
        self.numpoints =  numpoints
        if Domain != []:
            self.Domain = Domain
        elif self.numpoints>0:
            Domain=[]
            boundDomain=[]
            if self.dim==1:
                self.point1= point1
                self.point2 = point2
            elif self.dim==2:
##                print "setting self.Xaffine,self.Yaffine,self.Xgrid,self.Ygrid "
                X0 = point1[0]
                X1 = point2[0]
                Y0 = point1[1]
                Y1 = point2[1]
                self.Xaffine    = matrix([[X0, 1],[X1, 1]]).I*matrix([-1,1]).T
                self.Yaffine    = matrix([[Y0, 1],[Y1, 1]]).I*matrix([-1,1]).T
                self.Xgrid = numpy.linspace(X0,X1,self.numpoints)
                self.Ygrid = numpy.linspace(Y0,Y1,self.numpoints)
                ###print Xaffine, Yaffine
               # ##print len(Xgrid)
                for i in range(1,len(self.Ygrid)-1):
                    Domain+=[map(lambda x: [x,self.Ygrid[i]], self.Xgrid[1:len(self.Xgrid)-1])]
                for i in range(0,len(self.Ygrid)):
                    boundDomain+=[map(lambda x: [x,self.Ygrid[i]], self.Xgrid[0:len(self.Xgrid)])]

            ##print boundDomain
            self.clippedDomain = copy.deepcopy(Domain)
            self.Domain= flatten(Domain, levels=1)
            self.boundDomain = boundDomain
            self.numBoundDomain = self.numpoints**2
            self.point1=point1
            self.point2=point2
            self.hx = self.Xgrid[1] - self.Xgrid[0]
            self.hy = self.Ygrid[1] - self.Ygrid[0]
    #set the boundary points for rectangular orthogonal Domain, num_points - 1 divides 8
    def setBoundaryPoints(self,num_points):

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
        ###print theta
        #consider whether to include the first point twice
        end = len(self.boundDomain)-1
        start =0
        i=int(len(self.boundDomain[0])/2.0)
        j=end
        ###print 'start' , i,j, theta[0],num_points, sqrt(len(prob.boundDomain))
        ###print self.boundDomain
        self.d_Ohm_indices = []
        for n in range(num_points):
            ###print n,i,j, theta[n],prob.boundDomain[i][j],self.point1,self.point2
            self.d_Ohm+=[self.boundDomain[i][j]]
            self.d_Ohm_indices += [[i,j]]
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
           # ##print s[i], counter
            if s[i] == lb:
                counter+=1
            elif s[i] == rb:
                counter-=1
            i+=1
        return i

    def setSourceFunction(self,sourceFunction):
        self.sourceFunction = sourceFunction
        sourceFunction=sourceFunction.replace("*cos","*math.cos")
        sourceFunction=sourceFunction.replace("*sin","*math.sin")
        sourceFunction=sourceFunction.replace("*exp","*math.exp")
        sourceFunction=sourceFunction.replace("*log","*math.log")
        sourceFunction=sourceFunction.replace("pi",str(math.pi))
        self.SF = []
        #the SF as an array, structured without topology
        for i in range(len(self.Domain)):
            x = self.Domain[i][0]
            y = self.Domain[i][1]
            self.SF += [float(eval(sourceFunction))]
        ##print self.sourceFunction
        ###print self.SF

         #the SF as a 2d matrix, structured with topology
        self.SF2 = [] 
        for i in range(len(self.boundDomain)):
            row=[]
            for j in range(len(self.boundDomain[0])):
                x = self.boundDomain[i][j][0]
                y = self.boundDomain[i][j][1]
                row+=[float(eval(sourceFunction))]
            self.SF2 += [row] 
    #assumes we have set the differential equation
    #derive the source function from differential equation and solution information
    #or alternatively, set solution
    def deriveSourceFunction(self, sln):
        self.sln = sln
        #differentiate the sln based on the F_list
        x=Symbol('x')
        y=Symbol('y')
        ###print eval(sln)
        F=[]
        if self.dim==2:
            F =  self.dimLists(self.dim,4)
            F[0][0] = eval(sln)
        elif self.dim==1:
            F =[0,0,0,0]
            F[0] = eval(sln)
        #DE_aux = self.DE.replace("F", "Sln")
       # ##print F
        for i,n in enumerate(self.F_list):
            if self.dim==2:
                ###print n[0],n[1], eval(sln)
                F[n[0]][n[1]] = diff(diff(eval(sln),x,n[0]),y,n[1])
                #F[n[0]][n[1]] = diff(eval(sln),x,n[0],y,n[1])
                ###print F[n[0]][n[1]]
            elif self.dom==1:
                F[n[0]] = diff(eval(sln),x,n[0])
        self.sourceFunction = str(simplify(eval(self.DE)))
        sourceFunction=self.sourceFunction
        sourceFunction=sourceFunction.replace("*cos","*math.cos")
        sourceFunction=sourceFunction.replace("*sin","*math.sin")
        sourceFunction=sourceFunction.replace("*exp","*math.exp")
        sourceFunction=sourceFunction.replace("*log","*math.log")
        sourceFunction=sourceFunction.replace("pi",str(math.pi))
        self.SF = []
        #the SF as an array, structured without topology
        for i in range(len(self.Domain)):
            x = self.Domain[i][0]
            y = self.Domain[i][1]
            self.SF += [float(eval(sourceFunction))]
        ##print self.sourceFunction
        ###print self.SF

         #the SF as a 2d matrix, structured with topology
        self.SF2 = []
        self.Sln2 =[]
        for i in range(len(self.boundDomain)):
            row_srf=[]
            row_sln=[]
            for j in range(len(self.boundDomain[0])):
                x = self.boundDomain[i][j][0]
                y = self.boundDomain[i][j][1]
                row_srf+=[float(eval(sourceFunction))]
                row_sln+=[float(eval(sln))]
            self.SF2 += [row_srf]
            self.Sln2 += [row_sln]
                
    def setBoundarySegments(self):
        #circle_points = length(self.theta);
        self.d_Ohm_N = [];
        self.d_Ohm_D = [];
        self.d_Ohm_D_indices=[]
        self.d_Ohm_N_indices=[]
        j=0
        ###print len(self.theta)
        ###print self.Angles
        for i in range(len(self.theta)):
            angle = self.theta[i];
            
            if j < len(self.Angles)-1:
                b = self.Angles[j+1][1]
            else:
                b=2*pi
            if angle > b:
                j+=1
            if self.Angles[j][0] == "Dirichlet":
                self.d_Ohm_D += [self.d_Ohm[i]]
                self.d_Ohm[i] = [self.d_Ohm[i],"Dirichlet"]
                self.d_Ohm_D_indices+=[self.d_Ohm_indices[i]]
            elif self.Angles[j][0] == "Neumann":
               # ##print self.d_Ohm[i]
                self.d_Ohm_N += [self.d_Ohm[i]]
                self.d_Ohm[i] = [self.d_Ohm[i],"Neumann"]
                self.d_Ohm_N_indices+=[self.d_Ohm_indices[i]]

    ##To facilitate methods in higher level classes, such as ExtendBoundaries in McFall it
    ##is useful to treat each boundary point on the the boundary with an angle from 0 to 360.

    def setDirichletNeumannAngles(self, boundaryTypeId,numAngles=2,nstate = round(random.random())):
        self.Angles =[];
        self.boundaryTypeId=boundaryTypeId
        if boundaryTypeId == 0:
            self.Angles = [("Dirichlet",0)]
        elif boundaryTypeId == 1:
            self.Angles = [("Neumann",0)]     
        elif boundaryTypeId == 2:
            self.Angles = [("Dirichlet",0),("Neumann",0.25*2*pi), ("Dirichlet",0.75*2*pi)]
        elif boundaryTypeId == 3:
           numAngles = int(1+numAngles*round(random.random()))
          #print numAngles
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
        self.neumann_subset=[]
        #change to let the normals correspond to d_Ohm rather than d_Ohm_N
        for i,n in enumerate(self.d_Ohm):#map(lambda x: x[0],self.d_Ohm)):
            ###print i,n
##            if n[0]==self.point1[0] or n[0] ==self.point2[1]:
##                self.normal += [[1,0]]
##            elif n[1]==self.point1[1] or n[1] ==self.point2[1]:
##                self.normal += [[0,1]]
       # ##print self.normal
            t = n[1]
            n=n[0]
            if t=="Neumann":
                self.neumann_subset+=[i]
       
            if n[0]==self.point1[0]:
                self.normal += [[1,0]]
            elif n[0]==self.point2[0]:
                self.normal += [[-1,0]]
            elif n[1]==self.point1[1]:
                self.normal += [[0,1]]
            elif n[1]==self.point2[1]:
                self.normal += [[0,-1]]    

    ## creates the Dirichlet and Neumann boundary values on the boundary given the solution string
    def setBoundaryValues(self, sln):
        #dirichlet
        self.DC=[]
        sln_aux=self.markupString(sln)
        ###print sln
        for d in self.d_Ohm_D:
            x = d[0]
            y = d[1]
            self.DC+= [eval(sln_aux)]
##            print eval(sln_aux)
            ###print d, sln_aux, eval(sln_aux)
        ###print "Dirichlet", self.DC
        #neumann
        #obtain normalsself.
        self.setNormals()
        self.NC=[]
        ###print sln
        x=Symbol('x')
        y=Symbol('y')
        sln_x = self.markupString(str(simplify(diff(eval(sln),x))))
        sln_y = self.markupString(str(simplify(diff(eval(sln),y))))
##        print sln_x
##        print sln_y
        for i,n in enumerate(self.d_Ohm_N):
            ###print n
            x = n[0]
            y = n[1]
            self.NC+= [eval(sln_x)*self.normal[self.neumann_subset[i]][0]+eval(sln_y)*self.normal[self.neumann_subset[i]][1]]
##            print self.normal[self.neumann_subset[i]],(eval(sln_x),eval(sln_y)), eval(sln_x)*self.normal[self.neumann_subset[i]][0]+eval(sln_y)*self.normal[self.neumann_subset[i]][1]
##            print [self.normal[self.neumann_subset[i]][0],self.normal[self.neumann_subset[i]][1]]
       # ##print "Neumann", self.NC

            
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
        ##print DE
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
       # ##print self.F_list
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
                ###print Function_str[i]
                F[n[0]][n[1]] = Function(Function_str[i])
            elif self.dim ==1:
                F[n[0]] = Function(Function_str[i])
       # ##print DE_Aux
        D_p_string = str(diff(eval(DE_Aux),'param'))
       # ##print D_p_string, type(D_p_string)
        D_p_string=D_p_string.replace("(param)","")
       # ##print D_p_string, type(D_p_string)
        #remove (param)
        #find D index, identify l.h bracket and matching r.h bracket
        for j,n in enumerate(self.F_list):
            i=0
            #while i != -1:
            i = D_p_string.find("D", 0)
            start_i = i
            i+=2
           # ##print "start", start_i
            end_i = self.matchBracket(D_p_string, i, '(')
           # ##print "end", end_i
            fn_str = "F_p"
            for j,v in enumerate(n):
                fn_str += '[' + str(v) + ']'
            D_p_string= D_p_string[0:start_i] + fn_str+ D_p_string[end_i+1:len(D_p_string)]
           # ##print D_p_string
        D_p_string=self.markupString(D_p_string)
        self.setFunctions()
       # ##print D_p_string
        self.DE_p = D_p_string


 
