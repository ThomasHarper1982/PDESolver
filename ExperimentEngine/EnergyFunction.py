##EnergyFunction is a class that holds information pertaining to a problem - it is created in an application class
##and passed to Optimisation objects or methods. 

import numpy
from Solutions import *
from ProblemInformation import *
from numpy import asarray
import math
import copy

class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

def zeros(n,m):
    A= []
    for i in range(n):
        row=[]
        for j in range(m):
            row+=[0.0]
        A+=[row]
    return np.array(A)


#Energy Function is a static class that holds problem information and derives the function and jacobian values for
#a set of network parameters
class EnergyFunction:

    #load problem loads the Energy Function object with problem and representation information, also creates
    #static data structure used to increase efficiency in the function operation
    def loadProblem(probs, representation_type = "Implicit", basis_type = "Sigmoid",\
                    implicit_include_BC = True, mcfall=[]):
        print "loading Energy Function"
        EnergyFunction.prob_list = probs.prob_list
        EnergyFunction.prob = probs.prob_list[0]
        EnergyFunction.num_probs = len(probs.prob_list)
        EnergyFunction.representation_type = representation_type
        EnergyFunction.basis_type = basis_type
        EnergyFunction.mcfall = mcfall
        EnergyFunction.implicit_include_BC= implicit_include_BC
        EnergyFunction.DEError,EnergyFunction.DEP_Error,EnergyFunction.DirichletError,EnergyFunction.DirichletP_Error=[],[],[],[]
        EnergyFunction.NeumannError,EnergyFunction.NeumannP_Error,EnergyFunction.TotalError,EnergyFunction.TotalP_Error=[],[],[],[]
        EnergyFunction.ProblemTotal =zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))
        EnergyFunction.F=[]
        EnergyFunction.F_p=[]
        EnergyFunction.p_max = EnergyFunction.num_probs
        EnergyFunction.Errors = [0]*EnergyFunction.num_probs
        for p in range(EnergyFunction.num_probs):
            EnergyFunction.DEError += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.DEP_Error += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.DirichletError += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.DirichletP_Error += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.NeumannError += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.NeumannP_Error += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.TotalError += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]
            EnergyFunction.TotalP_Error += [zeros(len(EnergyFunction.prob.Xgrid),len(EnergyFunction.prob.Ygrid))]

            Domain=[]
            for i in range(len(EnergyFunction.prob.Xgrid)):
                Domain_row=[]
                for j in range(len(EnergyFunction.prob.Ygrid)):
                    k=3
                    Derivative=[]
                    for xn in range(4):
                        yn=0
                        Derivative_row=[]
                        while yn <= k:
                            Derivative_row+=[0]
                            yn+=1
                        Derivative+=[Derivative_row]
                        k-=1
                    Domain_row+=[Derivative]
                Domain+=[Domain_row]
            EnergyFunction.F+=[Domain]
                
            
            Domain=[]
            for i in range(len(EnergyFunction.prob.Xgrid)):
                Domain_row=[]
                for j in range(len(EnergyFunction.prob.Ygrid)):
                    k=3
                    Derivative=[]
                    for xn in range(4):
                        yn=0
                        Derivative_row=[]
                        while yn <= k:
                            Derivative_row+=[0]
                            yn+=1
                        Derivative+=[Derivative_row]
                        k-=1
                    Domain_row+=[Derivative]
                Domain+=[Domain_row]
            EnergyFunction.F_p+=[Domain]
                #generate FunctionList
        EnergyFunction.mixed = False
        if EnergyFunction.prob.boundaryTypeId > 0:
            EnergyFunction.mixed = True
        
##    EnergyFunction.SourceFunctions = map(lambda x:x.SF2,probs.prob_list)
##    EnergyFunction.DE = map(lambda x:x.SF2,probs.DE)
    
    loadProblem = Callable(loadProblem) 


    def unflatten(lst, cols):
        num_nodes = len(lst)/cols
        ma =[]
        k=0
        for j in range(cols):
            #col =[0]*num_nodes
            col=[]
            for i in range(num_nodes):
                col += [lst[k]]
                k+=1
            ma += [col]
        return array(ma)
    unflatten = Callable(unflatten)
    def Dirichlet(parameters, prob_info, deriv = '', param_index = -1,p=0):
       # print "bounded Domain: " , prob_info.boundDomain
        #print "calculating Dirichlet error"
        for i,d_Ohm_D in enumerate(EnergyFunction.prob.d_Ohm_D):
            indices = EnergyFunction.prob.d_Ohm_D_indices[i]
            #print "Dirichlet boundary :", d_Ohm_D, " indices : ", indices, " sln: ", prob_info.DC[i]
           # print "Dirichlet indices: " + str(indices)
            s = Solution.solution(0,0,indices[0],indices[1], d_Ohm_D, parameters,\
                                        basis_type=EnergyFunction.basis_type,\
                                        representation=EnergyFunction.representation_type, \
                                        mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                        param_derivative=deriv, param_index=param_index)
           # print "solution, target: ", s, prob_info.DC[i]
            if deriv == "":
                EnergyFunction.DirichletError[p][indices[0],indices[1]] = (s-prob_info.DC[i])
            else:
                EnergyFunction.DirichletP_Error[p][indices[0],indices[1]] = s

    Dirichlet = Callable(Dirichlet)
    def Neumann(parameters, prob_info,  deriv = '', param_index = -1,p=0):
       # print "calculating Neumann error"
        normal = EnergyFunction.prob.normal
        neumann_subset = EnergyFunction.prob.neumann_subset
        N = [0]*len(prob_info.d_Ohm_N)
##        print EnergyFunction.mcfall.Normal_ox
        for i,d_Ohm_N in enumerate(EnergyFunction.prob.d_Ohm_N):
            
            #neumann here!!
            dot_product = 0
            indices = EnergyFunction.prob.d_Ohm_N_indices[i]
##            print "\nNeumann indices: " + str(indices)
##            print "Neumann point ", d_Ohm_N
##            print "normal ", normal[neumann_subset[i]]
##            print "normal(v2) ", [EnergyFunction.mcfall.Normal_ox[indices[0]][indices[1]][0][0],\
##                                  EnergyFunction.mcfall.Normal_oy[indices[0]][indices[1]][0][0]]
            if normal[neumann_subset[i]][0]!=0:
                n_x = Solution.solution(1,0,indices[0],indices[1], d_Ohm_N, parameters,\
                                        basis_type=EnergyFunction.basis_type,\
                                        representation=EnergyFunction.representation_type, \
                                        mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                        param_derivative=deriv, param_index=param_index)
                dot_product = dot_product + n_x*normal[neumann_subset[i]][0]
            if normal[neumann_subset[i]][1]:
                n_y = Solution.solution(0,1,indices[0],indices[1], d_Ohm_N, parameters,\
                                        basis_type=EnergyFunction.basis_type,\
                                        representation=EnergyFunction.representation_type, \
                                        mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                        param_derivative=deriv, param_index=param_index)
                dot_product = dot_product + n_y*normal[neumann_subset[i]][1]
           # print "solution, target: ",dot_product,prob_info.NC[i]
            if deriv == "":
                EnergyFunction.NeumannError[p][indices[0],indices[1]] = (dot_product-prob_info.NC[i])
            else:
                EnergyFunction.NeumannP_Error[p][indices[0],indices[1]] = dot_product
 
    Neumann = Callable(Neumann)
    #return differential equation error, sum+=(D[f(x)] -g[x])^2
    def DifferentialEquation(param, prob, deriv = '', param_index = -1,p=0):
       # print "calulating DE error"

        offset = 1
        if EnergyFunction.implicit_include_BC==False and \
            EnergyFunction.representation_type == "implicit":
            offset = 0
        #print "offset" , offset
        error = 0
        Domain = prob.Domain
        if deriv == '':
            for i in range(offset,len(prob.Xgrid)-offset):
                for j in range(offset,len(prob.Ygrid)-offset):
                   # print i,j
                    X=prob.boundDomain[i][j]
                    DF = 0
                    for s in prob.F_list:
                       #print s
                        EnergyFunction.F[p][i][j][s[0]][s[1]] = Solution.solution(s[0],s[1],i,j,X,param, \
                                                                                  basis_type=EnergyFunction.basis_type,\
                                                                                  representation=EnergyFunction.representation_type, \
                                                                                  mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                    F= EnergyFunction.F[p][i][j]
                    x = X[0]
                    y = X[1]
                    DF = eval(prob.DE)
                    DF -= prob.SF2[i][j]
                    EnergyFunction.DEError[p][i,j] = DF
            if len(prob.d_Ohm_N)>0 and EnergyFunction.representation_type == "explicit":
                #we still need to solve the differential equation on the Neumann Boundary if the solution rep. is explicit
                for i,d_Ohm_N in enumerate(EnergyFunction.prob.d_Ohm_N):
                    indices = EnergyFunction.prob.d_Ohm_N_indices[i]
                    X=prob.boundDomain[indices[0]][indices[1]]
                    DF = 0
                    for s in prob.F_list:
                       #print s
                        EnergyFunction.F[p][indices[0]][indices[1]][s[0]][s[1]] = Solution.solution(s[0],s[1],indices[0],indices[1],X,param, \
                                                                                  basis_type=EnergyFunction.basis_type,\
                                                                                  representation=EnergyFunction.representation_type, \
                                                                                  mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                    F= EnergyFunction.F[p][indices[0]][indices[1]]
                    x = X[0]
                    y = X[1]
                    DF = eval(prob.DE)
                    DF -= prob.SF2[indices[0]][indices[1]]
                    EnergyFunction.DEError[p][indices[0],indices[1]] = DF
        else:
            for i in range(offset,len(prob.Xgrid)-offset):
                for j in range(offset,len(prob.Ygrid)-offset):
                    X=prob.boundDomain[i][j]
                    DF = 0
                    for s in prob.F_list:
                        EnergyFunction.F_p[p][i][j][s[0]][s[1]] = Solution.solution(s[0],s[1],i,j,X,param,\
                                                                                    basis_type=EnergyFunction.basis_type,\
                                                                                    representation=EnergyFunction.representation_type, \
                                                                                    param_derivative=deriv,param_index=param_index,\
                                                                                    mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                    F= EnergyFunction.F[p][i][j]
                    F_p= EnergyFunction.F_p[p][i][j]
                    x = X[0]
                    y = X[1]
                    DF = eval(prob.DE_p)
                    EnergyFunction.DEP_Error[p][i,j] = DF
            if len(prob.d_Ohm_N)>0 and EnergyFunction.representation_type == "explicit":
                #we still need to solve the differential equation on the Neumann Boundary if the solution rep. is explicit
                for i,d_Ohm_N in enumerate(EnergyFunction.prob.d_Ohm_N):
                    indices = EnergyFunction.prob.d_Ohm_N_indices[i]
                    X=prob.boundDomain[indices[0]][indices[1]]
                    DF = 0
                    for s in prob.F_list:
                        EnergyFunction.F_p[p][indices[0]][indices[1]][s[0]][s[1]] = Solution.solution(s[0],s[1],indices[0],indices[1],X,param,\
                                                                                    basis_type=EnergyFunction.basis_type,\
                                                                                    representation=EnergyFunction.representation_type, \
                                                                                    param_derivative=deriv,param_index=param_index,\
                                                                                    mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                    F= EnergyFunction.F[p][indices[0]][indices[1]]
                    F_p= EnergyFunction.F_p[p][indices[0]][indices[1]]
                    x = X[0]
                    y = X[1]
                    DF = eval(prob.DE_p)
                    EnergyFunction.DEP_Error[p][indices[0]][indices[1]] = DF
            
           # error += DF**2
        #add boundary component to error

    DifferentialEquation = Callable(DifferentialEquation)
        #return target mapping error
    #inputs parameter derivative
    
    def JacobianTM(parameters, prob_info):
        error = [0]*len(parameters)
        return error
    JacobianTM = Callable(JacobianTM)

    def JacobianDE(parameters, prob_info):
        error = [0]*len(parameters*(dim+2))
        return error
    JacobianDE = Callable(JacobianTM)

    #functions iterates over a list of ProblemInformation objects, summing the energy functions together
    def functions(param,  deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0):
        #print "energy function"
        prob_error = 0
        for p in range(EnergyFunction.p_max):
           # print "problem: ", p
            prob_error += EnergyFunction.function(param, EnergyFunction.prob_list[p], deriv=deriv, param_index=param_index,DE_inner=DE_inner, \
                                                  B_D_inner=B_D_inner, Neu_inner = Neu_inner,Target_inner=Target_inner,p=p)

        return prob_error
    functions = Callable(functions)
    ##function takes the parameters, assumes the correct problem/representation information is loaded, calculates and returns the error
    ##also constructs the 3d residual plot over the domain (EnergyFunction.Total_Error)
    ##the override allows explicit representation to be optimised with the BC in the energy function, may not be necessary
    def function(param, prob, deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0,p=0,overide=False):
##    '''
##    param- [alpha^i, beta^i, omega_1^i, omega_2^i], with i from 0 to n
##    prob - the ProblemInformation object
##    deriv - partially differentiates with respect to 'alpha', 'beta', 'omega1' or 'omega2'
##    param_index - the node of the parameter being differentiated
##    '''
      #  print "after ", parameters
        error = 0
        N_weight = 0
        D_weight=10
        #no parameter derivative
        if deriv== "":
            EnergyFunction.DifferentialEquation(param, prob,p=p)
            #print "DE:",EnergyFunction.DEError[p]
            EnergyFunction.TotalError[p] = EnergyFunction.DEError[p]**2#sum(map(lambda x: x**2, DE))/len(prob_info.Domain)
            #print "DE**2:",EnergyFunction.TotalError[p]
            if EnergyFunction.implicit_include_BC==True and \
               EnergyFunction.representation_type == "implicit" or overide:
                EnergyFunction.Dirichlet(param, prob,p=p)
                EnergyFunction.TotalError[p]+= D_weight*EnergyFunction.DirichletError[p]**2#sum(map(lambda x: x**2, B_D))/len(prob_info.d_Ohm_D)
              #  print "total dirichlet: ", D_weight*EnergyFunction.DirichletError[p]**2
                if len(prob.d_Ohm_N) > 0:
                    EnergyFunction.Neumann(param, prob,p=p)                     
                    EnergyFunction.TotalError[p]+= N_weight*EnergyFunction.NeumannError[p]**2#sum(map(lambda x: x**2, B_N))/len(prob_info.d_Ohm_N)
            error = numpy.sum(EnergyFunction.TotalError[p])/len(flatten(prob.boundDomain,levels=1))#(len(prob.boundDomain)*len(prob.boundDomain[0])) 
            EnergyFunction.Errors[p] = numpy.sum(EnergyFunction.TotalError[p])/len(flatten(prob.boundDomain,levels=1))
        else:
        #parameter derivative
            if DE_inner == 0:
                EnergyFunction.DifferentialEquation(param, prob,p=p)
            EnergyFunction.DifferentialEquation(param, prob, deriv, param_index,p=p)
            
            EnergyFunction.TotalP_Error[p] = EnergyFunction.DEError[p]*EnergyFunction.DEP_Error[p] # =2*sum(DE_inner*DE_p)/len(prob_info.Domain)
           # print DE_inner, "**", DE_p, "**", error
            #print parameters, deriv, param_index
            if EnergyFunction.implicit_include_BC==True and \
               EnergyFunction.representation_type == "implicit" or overide:
                if B_D_inner==0:
                    EnergyFunction.Dirichlet(param, prob,p=p)
                EnergyFunction.Dirichlet(param, prob,deriv, param_index,p=p)
               # print B_D_inner, EnergyFunction.DirichletError[p]
                EnergyFunction.TotalP_Error[p] += 2*D_weight*EnergyFunction.DirichletP_Error[p]*\
                                                  EnergyFunction.DirichletError[p]#2*D_weight*sum(B_D_inner*B_D_p)/len(B_D_p)
                #print B_D_inner, "**", B_D_p, "**", error
                if len(prob.d_Ohm_N) > 0:
                    EnergyFunction.Neumann(param, prob,p=p)
                    #print N_weight, Neu_inner,EnergyFunction.NeumannP_Error[p]
                    EnergyFunction.TotalP_Error[p] += 2*N_weight*EnergyFunction.NeumannP_Error[p]*EnergyFunction.NeumannError[p]
            
            error = numpy.sum(EnergyFunction.TotalP_Error[p])/len(flatten(prob.boundDomain,levels=1))#(len(prob.boundDomain)*len(prob.boundDomain[0]))
        return error
    function = Callable(function)


    #supplies a one to one correspondence of D_param with the param                                               
    def jacobians(param):
        prob_error = numpy.array([0.0]*len(param))
        for p in range(EnergyFunction.p_max):
            prob_error += EnergyFunction.jacobian(param, EnergyFunction.prob_list[p],p=p)
        return np.array(prob_error)
    jacobians = Callable(jacobians)
                                                 
    def jacobian(param,prob,p=0,overide=False):
        #  print "jacobian"
        # print parameters
        #  print type(parameters)
        EnergyFunction.DifferentialEquation(param,prob,p=p)
        #B_D_inner, Neu_inner = 0,0
        if EnergyFunction.implicit_include_BC==True and \
           EnergyFunction.representation_type == "implicit" or overide:
            EnergyFunction.Dirichlet(param,prob,p=p)#B_D_inner = 
            if len(prob.d_Ohm_N) > 0:
                EnergyFunction.Neumann(param, prob,p=p)
        #print DE_inner, B_D_inner
        param_deriv = ["alpha","beta","omega1","omega2"]
##        der = numpy.zeros_like([0.0]*((prob_info.dim+2)*len(parameters[0,:].T)))
##        der = numpy.array()
        der = []
        for param_index in range(numpy.size(param)/4):
            for j,deriv in enumerate(param_deriv):
                der += [EnergyFunction.function(param, prob, deriv=deriv, param_index=param_index, p=p)]                           
        return numpy.array(der)                                       
    jacobian = Callable(jacobian) 
    def viewError(p=0):
        fig = plt.figure()
        fig.suptitle('plot of error residuals squared')
        ax = fig.gca(projection='3d')

        #sum the errors for each problem
        
        for p in range(EnergyFunction.num_probs):
            EnergyFunction.ProblemTotal += EnergyFunction.TotalError[p]

        X,Y = np.meshgrid(EnergyFunction.prob.Xgrid, EnergyFunction.prob.Ygrid)
       # print EnergyFunction.ProblemTotal
        surf = ax.plot_surface(X,Y,EnergyFunction.ProblemTotal ,rstride=1, cstride=1, cmap=cm.jet,
                               linewidth=0, antialiased=False)

        minV = EnergyFunction.ProblemTotal.min()
        maxV = EnergyFunction.ProblemTotal.max()
        function_height = maxV-minV
        ax.set_zlim3d(minV-0.2*function_height, maxV+0.2*function_height)
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(surf, shrink=0.5,aspect=5)
        plt.show()

    viewError = Callable(viewError) 

    def truncated_functions(param,  deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0,limit=float('inf')):
        #print "energy function"
        prob_error = 0
        for p in range(EnergyFunction.p_max):
           # print "problem: ", p
            prob_error += EnergyFunction.truncated_function(param, EnergyFunction.prob_list[p], deriv=deriv, param_index=param_index,DE_inner=DE_inner, \
                                                  B_D_inner=B_D_inner, Neu_inner = Neu_inner,Target_inner=Target_inner,p=p,limit=limit)
            
            if prob_error > limit:
                return float(str('inf'))
        return prob_error
    truncated_functions = Callable(truncated_functions)

    #TRUNCATED energy function, for non-differentiable optimisation where there is a error bound, stops
    #calculating beyond this error and returns inifinity, has
    #a different topology to the differentiable energy function, we cannot simply encapsulate certain functions
    def truncated_function(param, prob, deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0,p=0,overide=False,limit=float('inf')):        
      #  print "after ", parameters
        error = 0
        N_weight = 0
        D_weight=10
        #no parameter derivative

       #the offset refers to the index at the boundary, takes either side of the boundary
        offset = 1
        if EnergyFunction.implicit_include_BC==False and \
            EnergyFunction.representation_type == "implicit":
            offset = 0
        ##evaluate the differential equations
        for i in range(offset,len(prob.Xgrid)-offset):
            for j in range(offset,len(prob.Ygrid)-offset):
               # print i,j
                X=prob.boundDomain[i][j]
                DF = 0
                for s in prob.F_list:
                   #print s
                    EnergyFunction.F[p][i][j][s[0]][s[1]] = Solution.solution(s[0],s[1],i,j,X,param, \
                                                                              basis_type=EnergyFunction.basis_type,\
                                                                              representation=EnergyFunction.representation_type, \
                                                                              mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                F= EnergyFunction.F[p][i][j]
                x = X[0]
                y = X[1]
                DF = eval(prob.DE)
                DF -= prob.SF2[i][j]
                EnergyFunction.DEError[p][i,j] = DF
                error += (EnergyFunction.DEError[p][i,j]**2)/len(flatten(prob.boundDomain,levels=1))
                if error > limit:
                    return float(str('inf'))
        if len(prob.d_Ohm_N)>0 and EnergyFunction.representation_type == "explicit":
            #we still need to solve the differential equation on the Neumann Boundary if the solution rep. is explicit
            for i,d_Ohm_N in enumerate(EnergyFunction.prob.d_Ohm_N):
                indices = EnergyFunction.prob.d_Ohm_N_indices[i]
                X=prob.boundDomain[indices[0]][indices[1]]
                DF = 0
                for s in prob.F_list:
                    EnergyFunction.F_p[p][indices[0]][indices[1]][s[0]][s[1]] = Solution.solution(s[0],s[1],indices[0],indices[1],X,param,\
                                                                                basis_type=EnergyFunction.basis_type,\
                                                                                representation=EnergyFunction.representation_type, \
                                                                                param_derivative=deriv,param_index=param_index,\
                                                                                mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed)
                F= EnergyFunction.F[p][indices[0]][indices[1]]
                F_p= EnergyFunction.F_p[p][indices[0]][indices[1]]
                x = X[0]
                y = X[1]
                DF = eval(prob.DE_p)
                EnergyFunction.DEP_Error[p][indices[0]][indices[1]] = DF
        if EnergyFunction.implicit_include_BC==True and \
           EnergyFunction.representation_type == "implicit" or overide:
            #evaluate the Dirichlet term
            for i,d_Ohm_D in enumerate(EnergyFunction.prob.d_Ohm_D):
                indices = EnergyFunction.prob.d_Ohm_D_indices[i]
               # print "Dirichlet indices: " + str(indices)
                s = Solution.solution(0,0,indices[0],indices[1], d_Ohm_D, param,\
                                            basis_type=EnergyFunction.basis_type,\
                                            representation=EnergyFunction.representation_type, \
                                            mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                            param_derivative=deriv, param_index=param_index)
                EnergyFunction.DirichletError[p][indices[0],indices[1]] = D_weight*((s-prob.DC[i])**2)/len(flatten(prob.boundDomain,levels=1))
                error+=EnergyFunction.DirichletError[p][indices[0],indices[1]]
                if error > limit:
                    return float(str('inf'))
            if len(prob.d_Ohm_N) > 0:
                normal = EnergyFunction.prob.normal
                neumann_subset = EnergyFunction.prob.neumann_subset
                N = [0]*len(prob.d_Ohm_N)
    ##        print EnergyFunction.mcfall.Normal_ox
                for i,d_Ohm_N in enumerate(EnergyFunction.prob.d_Ohm_N):
                    
                    #neumann here!!
                    dot_product = 0
                    indices = EnergyFunction.prob.d_Ohm_N_indices[i]

                    if normal[neumann_subset[i]][0]!=0:
                        n_x = Solution.solution(1,0,indices[0],indices[1], d_Ohm_N, param,\
                                                basis_type=EnergyFunction.basis_type,\
                                                representation=EnergyFunction.representation_type, \
                                                mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                                param_derivative=deriv, param_index=param_index)
                        dot_product = dot_product + n_x*normal[neumann_subset[i]][0]
                    if normal[neumann_subset[i]][1]:
                        n_y = Solution.solution(0,1,indices[0],indices[1], d_Ohm_N, param,\
                                                basis_type=EnergyFunction.basis_type,\
                                                representation=EnergyFunction.representation_type, \
                                                mcfall=EnergyFunction.mcfall, mixed=EnergyFunction.mixed,\
                                                param_derivative=deriv, param_index=param_index)
                        dot_product = dot_product + n_y*normal[neumann_subset[i]][1]
                   # print "solution, target: ",dot_product,prob_info.NC[i]
                    
                    EnergyFunction.NeumannError[p][indices[0],indices[1]] = (dot_product-prob.NC[i])
                    error+=(EnergyFunction.NeumannError[p][indices[0],indices[1]]**2)/len(flatten(prob.boundDomain,levels=1))
                    if error > limit:
                        return float(str('inf'))
        return error
    truncated_function = Callable(truncated_function)

    #to be used with the Lavenburg Marquardt algorithm, leastsq
    #functions iterates over a list of ProblemInformation objects, accumulating residuals together
    def functions_diff(param,  deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0):
        #print "energy function"
        prob_error = numpy.array([])
        for p in range(EnergyFunction.p_max):
           # print "problem: ", p
            prob_error  = append(prob_error, EnergyFunction.function_diff(param, EnergyFunction.prob_list[p], deriv=deriv, param_index=param_index,DE_inner=DE_inner, \
                                                  B_D_inner=B_D_inner, Neu_inner = Neu_inner,Target_inner=Target_inner,p=p),1)
        #print "differences :" ,prob_error
        return prob_error
    functions_diff = Callable(functions_diff)
    ##function takes the parameters, assumes the correct problem/representation information is loaded, calculates and returns the error
    ##the override allows explicit representation to be optimised with the BC in the energy function, may not be necessary
    def function_diff(param, prob, deriv="", param_index=-1,DE_inner=0, \
                 B_D_inner=0, Neu_inner = 0,Target_inner=0,p=0,overide=False):        
      #  print "after ", parameters
        error = 0
        N_weight = 0
        D_weight=10
        #no parameter derivative
        error = numpy.array([])
        if deriv== "":
            EnergyFunction.DifferentialEquation(param, prob,p=p)

            error = concatenate((error, EnergyFunction.DEError[p].flatten()),axis=1)
            if EnergyFunction.implicit_include_BC==True and \
               EnergyFunction.representation_type == "implicit" or overide:
               # print "Dirichlet boundary"
                EnergyFunction.Dirichlet(param, prob,p=p)
                #EnergyFunction.TotalError[p]+= D_weight*EnergyFunction.DirichletError[p]**2#sum(map(lambda x: x**2, B_D))/len(prob_info.d_Ohm_D)
                error = concatenate((error, EnergyFunction.DirichletError[p].flatten()),axis=1)          
              #  print "total dirichlet: ", D_weight*EnergyFunction.DirichletError[p]**2
                if len(prob.d_Ohm_N) > 0:
                    EnergyFunction.Neumann(param, prob,p=p)                     
                   # EnergyFunction.TotalError[p]+= N_weight*EnergyFunction.NeumannError[p]**2#sum(map(lambda x: x**2, B_N))/len(prob_info.d_Ohm_N)
                    error = concatenate((error, EnergyFunction.NeumannError[p].flatten()),axis=1) 
            error = error/len(flatten(prob.boundDomain,levels=1))
            #error = numpy.sum(EnergyFunction.TotalError[p]))#(len(prob.boundDomain)*len(prob.boundDomain[0])) )
        else:
        #parameter derivative
            EnergyFunction.DifferentialEquation(param, prob, deriv, param_index,p=p)
            
            error = concatenate((error,(EnergyFunction.DEError[p]*EnergyFunction.DEP_Error[p]).flatten()),axis=1) # =2*sum(DE_inner*DE_p)/len(prob_info.Domain)

            if EnergyFunction.implicit_include_BC==True and \
               EnergyFunction.representation_type == "implicit" or overide:
##                if B_D_inner==0:
##                    EnergyFunction.Dirichlet(param, prob,p=p)

              
                EnergyFunction.Dirichlet(param, prob,deriv=deriv, param_index=param_index, p=p)
               # print B_D_inner, EnergyFunction.DirichletError[p]
                error = concatenate((error,(EnergyFunction.DirichletP_Error[p]* EnergyFunction.DirichletError[p]).flatten()),axis=1)

                if len(prob.d_Ohm_N) > 0:
                                
                    EnergyFunction.Neumann(param, prob,deriv=deriv, param_index=param_index, p=p)
                    #print N_weight, Neu_inner,EnergyFunction.NeumannP_Error[p]
                    error = concatenate((error,(Neu_inner*EnergyFunction.NeumannP_Error[p]*EnergyFunction.NeumannError[p]).flatten()),axis=1)
                   # print "Neumann ",(Neu_inner*EnergyFunction.NeumannP_Error[p]*EnergyFunction.NeumannError[p]).flatten()
            error = error/len(flatten(prob.boundDomain,levels=1))#(len(prob.boundDomain)*len(prob.boundDomain[0]))
        return error

    ##jacobian of each residual of the energy function
    ##derivatives across rows, residuals across columns
                                            
    def jacobians_residual(param):
        if EnergyFunction.p_max>0:
            prob_error =EnergyFunction.jacobian_residual(param, EnergyFunction.prob_list[0],p=0)
        for p in range(1,EnergyFunction.p_max):
            prob_error = append(prob_error,EnergyFunction.jacobian_residual(param, EnergyFunction.prob_list[p],p=p),0)
        return np.array(prob_error)
    jacobians_residual = Callable(jacobians_residual)
                                                 
    def jacobian_residual(param,prob,p=0,overide=False):
        #  print "jacobian"
        # print parameters
        #  print type(parameters)
       # print "calculating outer function"
        DE_inner=EnergyFunction.DifferentialEquation(param,prob,p=p)
        #B_D_inner, Neu_inner = 0,0
        if EnergyFunction.implicit_include_BC==True and \
           EnergyFunction.representation_type == "implicit" or overide:
            EnergyFunction.Dirichlet(param,prob,p=p)#B_D_inner = 
            if len(prob.d_Ohm_N) > 0:
                EnergyFunction.Neumann(param, prob,p=p)
        #print DE_inner, B_D_inner
        param_deriv = ["alpha","beta","omega1","omega2"]
##        der = numpy.zeros_like([0.0]*((prob_info.dim+2)*len(parameters[0,:].T)))
##        der = numpy.array()
        der = asmatrix(EnergyFunction.function_diff(param, prob, deriv=param_deriv[0], param_index=0, p=p))
       # print "calcuating inner derivatives"
        for param_index in range(numpy.size(param)/4):
            for j,deriv in enumerate(param_deriv):
               # print param_index,deriv
                if deriv!=param_deriv[0] or param_index!=0:
                    row = asmatrix(EnergyFunction.function_diff(param, prob, deriv=deriv, param_index=param_index, p=p))
                   # print "row", row
                    der = concatenate((der,row),axis=0)
                #print "derivatives,residuals", der
        return der                                      
    jacobian_residual = Callable(jacobian_residual) 
    
    function_diff = Callable(function_diff)

    
    
