##A collection of methods used in optimisation
##A optimisation object that arranges the methods into sequence

import numpy
from optimize import fmin_bfgs, fmin_cg
from optimize import fmin
import scipy
from scipy.optimize import *
from Network import *
import random
import copy
import math

class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

def randomParams(param):
    for i in range(len(param)):
        param[i] = 5-10*random.random()
    return param
def zeros(n,m):
    A= []
    for i in range(n):
        row=[]
        for j in range(m):
            row+=[0.0]
        A+=[row]
    return np.array(A)
def infinities(n,m):
    A= []
    for i in range(n):
        row=[]
        for j in range(m):
            row+=[float('inf')]
        A+=[row]
    return np.array(A)

def permute(A,P):
    B=[0]*len(A)
    for i,a in enumerate(A):
        B[i] =A[P[i]]
    return B

def argmin(A):
    min_val = float('inf')
    min_index = 0
    for i in range(len(A)):
        if A[i] < min_val:
            min_val = A[i]
            min_index = i
    return (min_index,min_val)
def flatten(*args):
    x = []
    for l in args:
        if not isinstance(l, (list, tuple)): l = [l]
        for item in l:
            if isinstance(item, (list,tuple)):
                x.extend(flatten(item))
            else:
                x.append(item)
    return x

#determine truncated error
#if error is infinity, keep trying above up to maxfails
#   and if so then increment bucketNum, set fails and L to
#else
#add to existing bucket
#set 

def MonteCarlo(EnergyFunction,centre, radius, numSolutions, MaxNumFails=40):
    buckets = []
    currentbucket=[]
    bucketError=[float('inf')]
    bucketNum =0
    i=0
    fails=0
    L = float('inf')
    while i < numSolutions:
        #generate new parameters
        
        randnums =numpy.array(map(lambda x: 1-2*random.random(), centre))
        param = centre + radius*randnums
        
##        param=[]
##        for j in range(len(centre)/4):
##            omega1=(-1)**(-round(random.random()))*10**((-0.25+2*(1-2*random.random())))
##            omega2=(-1)**(-round(random.random()))*10**((-0.25+2*(1-2*random.random())))
##            alpha=(-1)**(-round(random.random()))*10**((0.25+1.5*(1-2*random.random())))
##            beta = (-1)**(-round(random.random()))*10**((-0.25+2*(1-2*random.random())))
##            param += [alpha,beta,omega1,omega2]
##        param=numpy.array(param)
        error =  EnergyFunction.truncated_functions(param,limit=L)
        if error == float('inf'):
            fails+=1
            if fails == MaxNumFails:
                fails=0
                L = float('inf')
                bucketNum+=1
                buckets+=[currentbucket]
                currentbucket=[]
        else:
            print "solution " +str(i) +" error: " +  str(error)
            
            currentbucket = [Sol(param,error)] + currentbucket
            L=error
            i+=1
    buckets+=[currentbucket]
    solution_list=flatten(buckets)
    solution_list.sort(key=lambda x:x.error)
    print map(lambda x:x.error,solution_list)
    return solution_list

#take the contribution of the ith node where 0<=i<=len(param)/4
def Contribution(EnergyFunction,param, i):
    param_aux = copy.deepcopy(param)
    param_aux[i*4:i*4+4] = numpy.array([0,0,0,0])
    return EnergyFunction.truncated_functions(param_aux)-EnergyFunction.truncated_functions(param)

def OmitZeroNodes(EnergyFunction,param):
    zero = 0.00001
    p11=EnergyFunction.prob.point1[0]
    p12=EnergyFunction.prob.point1[1]
    p21=EnergyFunction.prob.point2[0]
    p22=EnergyFunction.prob.point2[1]
    point1 = EnergyFunction.prob.point1
    point2 = EnergyFunction.prob.point2
    point3 = [EnergyFunction.prob.point1[0],EnergyFunction.prob.point2[1]]
    point4 = [EnergyFunction.prob.point2[0],EnergyFunction.prob.point1[1]]
    zero_nodes=[]
    for i in range(len(param)/4):
        f_1 = Network.function(0,0,"",point1,param[i*4:i*4+4],EnergyFunction.basis_type)
        f_2 = Network.function(0,0,"",point1,param[i*4:i*4+4],EnergyFunction.basis_type)
        f_3 = Network.function(0,0,"",point1,param[i*4:i*4+4],EnergyFunction.basis_type)
        f_4 = Network.function(0,0,"",point1,param[i*4:i*4+4],EnergyFunction.basis_type)
        if f_1<zero and f_2 <zero and f_3<zero and f_4<zero:
            param[i*4:i*4+4]= numpy.array([0,0,0,0])
            zero_nodes+=[i]
    print "omit: ", zero_nodes
    return (param,zero_nodes)


def OmitNegativeNodes(EnergyFunction,param):
    minContribution = -1
    zero_nodes=[]
    while minContribution<0:
        Cont_list=[]
        for i in range(len(param)/4):
            Cont_list+=[Contribution(EnergyFunction,param, i)]
        print "Contributions: " + str(Cont_list)
        (j,minContribution)=argmin(Cont_list)
        print "smallest: index, minContribution " +str(j) + ", "+ str(minContribution)
        if minContribution<0:
            param[j*4:j*4+4] = numpy.array([0,0,0,0])
            zero_nodes+=[j]
    return (param,zero_nodes)

def SplitNodes(EnergyFunction,param,zero_nodes):
    num = len(param)/4
    if len(zero_nodes) < len(param):
        for z in zero_nodes:
            i = z
            while i in zero_nodes:
                i = int(num*random.random())
            a=random.random()
            param[4*z:4*z+4]=copy.deepcopy(param[4*i:4*i+4])
            alpha = param[4*i]
            param[4*z] =  a*alpha
            param[4*i] =  (1-a)*alpha
    return param

def largeWeights(param,i,freq):
    if i%freq==0:
        num = len(param)/4
        for i in range(num):
            if abs(param[4*i+1]) > 40 or abs(param[4*i+2]) >40 or abs(param[4*i+3])>40:
                return True
    return False

def bound(velocities,max_vel):
    #num =  len(velocities)/4
    for i in range(len(velocities)):
        j=i%4    
        if velocities[i]  < -1*max_vel[j]:  
            velocities[i] = -1*max_vel[j]
        elif max_vel[j] < velocities[i]:
            velocities[i] = max_vel[j]

def proximalInitialSolutions(EnergyFunction,num_solutions, num_nodes, MaxNumFails=5):
    #use the boundary to derive alpha range
    #work out a series of random points in the domain for each node
    #change the levels of randomness for omega1,omega2 for each node
    #for each node/random_point (rand_sign)*10^x where x=[-2.25,-0.25,1.75]
    #work out random parameters for omega1,omega2
    #derive the beta so the inflection cuts through the randompoint
    #ie: omega1*x0 +omega2*y0 => beta
    #derive alpha from the boundary 2*(max-min)
    random_points=[]
    point1=EnergyFunction.prob.point1
    point2=EnergyFunction.prob.point2
    for i in range(num_nodes):
        x=point1[0] + random.random()*(point2[0]-point1[0])
        y=point1[1] + random.random()*(point2[1]-point1[1])
        random_points+=[[x,y]]
    #monte-carlo style improvement
    buckets = []
    currentbucket=[]
    bucketError=[float('inf')]
    bucketNum =0
    i=0
    fails=0
    L = float('inf')
    while i < num_solutions:
        #generate new parameters
        
        #randnums =numpy.array(map(lambda x: 1-2*random.random(), centre))
        #param = centre + radius*randnums
        param=[]
        for j in range(num_nodes):
            omega1=(-1)**(-round(random.random()))*30**(((1-2*random.random())))
            omega2=(-1)**(-round(random.random()))*30**(((1-2*random.random())))
            alpha=(-1)**(-round(random.random()))*10**(((1-2*random.random())))
            beta = -(omega1*random_points[j][0] + omega2*random_points[j][1])
            #beta = -(omega1*random.random()*(point2[0]-point1[0]) + omega2*random.random()*(point2[1]-point1[1]))
            
            #beta = beta*(1+0.1*(1-2*random.random()))
            param += [alpha,beta,omega1,omega2]
        param = numpy.array(param)
        error =  EnergyFunction.truncated_functions(param,limit=L)
        if error == float('inf'):
            fails+=1
            if fails == MaxNumFails:
                fails=0
                L = float('inf')
                bucketNum+=1
                buckets+=[currentbucket]
                currentbucket=[]
        else:
            print "solution " +str(i) +" error: " +  str(error)
            
            currentbucket = [Sol(param,error)] + currentbucket
            L=error
            i+=1
    buckets+=[currentbucket]
    solution_list=flatten(buckets)
    solution_list.sort(key=lambda x:x.error)
    print map(lambda x:x.error,solution_list)
    return solution_list
    
        
def ParticleSwarmOptimisation(EnergyFunction,particle_list,
                              MaxIterations=30,ErrorThreshold=1e-4):
    c1=0.5
    c2=1
    decay = 0.95
    momentum=0.9
    mutation_rate=0.001
    for p in particle_list:
        p.local_best_error = copy.deepcopy(p.error)
        p.local_best_param = copy.deepcopy(p.param)
        p.velocity = numpy.array([0]*len(p.param))
        p.velocity_old =  numpy.array([0]*len(p.param))
    globalBestParticle = 0
    globalBestError = particle_list[globalBestParticle].error
    it=0
    while it< MaxIterations and globalBestError>ErrorThreshold:
        print "iteration :" + str(it) + " globalBestError: "+ str(globalBestError)
        for i,p in enumerate(particle_list):
####            p.param[4*0+1]=-10000
##            p.param[4*0+2]=10000
##            p.param[4*0+3]=10000
            #print p.param
            #print p.velocity
            error =  EnergyFunction.truncated_functions(p.param,limit=p.local_best_error)
            if error < p.local_best_error:
                    p.local_best_error = error
                    p.local_best_param= p.param
                    #if largeWeights(p.local_best_param,i,5):
##                       # print 'large weights'
##                        #omit negative noces
##                        #split zero nodes
                        #(p.param,zero_nodes)=OmitNegativeNodes(EnergyFunction,p.local_best_param)
                       # (p.param,zero_nodes)=OmitZeroNodes(EnergyFunction,p.local_best_param)
##                        p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param,limit=float('inf'))
####                        print 'error before splitting nodes',error
####                        print 'params before',p.param
##                        p.param = SplitNodes(EnergyFunction,p.local_best_param,zero_nodes)
####                        p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param,limit=float('inf'))
####                        print 'error after splitting nodes',error
####                        print 'params after',p.param 
                        
            if p.local_best_error < globalBestError:
                globalBestError = p.local_best_error
                globalBestParticle = i
        for i,p in enumerate(particle_list):
            R = random.random()
            if R > mutation_rate: 
                lambda1 = numpy.array(map(lambda x:1-2*random.random(), p.param))
                lambda2 = numpy.array(map(lambda x:1-2*random.random(), p.param))
                globalBest = particle_list[globalBestParticle].local_best_param
                p.velocity = c1*lambda1*(p.local_best_param-p.param) + c2*lambda2*(globalBest-p.param)
                #bound(p.velocity,[100,5,5,5])
                p.param = p.param +p.velocity + momentum*p.velocity_old
                p.velocity_old = copy.deepcopy(p.velocity)
                momentum = momentum*decay
            else:
                print "mutate particle ",i, " with error ", str(p.local_best_error), " best particle ", globalBestParticle
                scale = 0.5
                lamb = numpy.array(map(lambda x:1-2*random.random(), p.param))
                p.param += lamb*scale*p.param
                scale = 0.1
                lamb = numpy.array(map(lambda x:1-2*random.random(), p.local_best_param))
                p.local_best_param += lamb*scale*p.local_best_param
                p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param)
                p.velocity_old = 0*p.velocity_old

                globalBestError = float('inf')
                globalBestParticle = -1
                for i1,p1 in enumerate(particle_list):
                    if p1.local_best_error < globalBestError:
                        globalBestParticle=i1
                        globalBestError = p1.local_best_error
                #determine global best error
##                R =random.random()    
##            if  R < 0.05:
##                print "random chance"
##                #choose 5 random particles, not the best
##                ran_particles = []
##                for j in  range(5):
##                    p = int(random.random()*len(particle_list))
##                    if p != globalBestParticle:
##                        ran_particles += [p]
##                for i in range(5):
##                    p = particle_list[i]
##                    p.param =  particle_list[globalBestParticle].local_best_param*(1+0.1*(1-2*random.random()))
##                    p.local_best_param =  particle_list[globalBestParticle].local_best_param*(1+0.1*(1-2*random.random()))
##                    p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param)
##                    if p.local_best_error < globalBestError:
##                        globalBestParticle=i
##                        globalBestError = p.local_best_error
        it+=1
    #sort particles in order
    particle_list.sort(key=lambda x:x.local_best_error)
##    print "errors ", map(lambda x:x.local_best_error,particle_list)
    for p in particle_list:
        p.error = copy.deepcopy(p.local_best_error)
        p.param = copy.deepcopy(p.local_best_param)
    return particle_list

#this method uses Praveen Koduru, Sanjoy Das, Stephen M. Welch method from their papaer A Particle Swarm Optimization-Nelder Mead Hybrid
##Algorithm for Balanced Exploration and Exploitation in Multidimensional Search Space 

def PSO_NM_hyrid(EnergyFunction,particle_list,
                    MaxIterations=10,ErrorThreshold=1e-4):
    c1=0.5
    c2=1
    decay = 0.95
    momentum=0.9

    k= 5
    cluster_centers=[]
    cluster_errors=[0]*k
    #initialise k  random clustering centers
    for i in range(k):
        cluster_centers += [Sol(numpy.array(map(lambda x: 1-2*random.random(),
                                                [20,5,2,2]*(len(particle_list[0].param)/4))),-1)]
        #cluster_errors += [EnergyFunction.truncated_functions(cluster_centers[i],limit=p.local_best_error)]
    for p in particle_list:
        p.local_best_error = copy.deepcopy(p.error)
        p.local_best_param = copy.deepcopy(p.param)
        p.velocity = numpy.array([0]*len(p.param))
        p.velocity_old =  numpy.array([0]*len(p.param))
    BestParticle = 0
    BestError = particle_list[BestParticle].error
    it=0
    while it< MaxIterations and BestError>ErrorThreshold:
        print "iteration :" + str(it) + " BestError: "+ str(BestError)
        for i,p in enumerate(particle_list):
####            p.param[4*0+1]=-10000
##            p.param[4*0+2]=10000
##            p.param[4*0+3]=10000
            #print p.param
            #print p.velocity
            error =  EnergyFunction.truncated_functions(p.param,limit=p.local_best_error)
            if error < p.local_best_error:
                    p.local_best_error = error
                    p.local_best_param= p.param
                    if largeWeights(p.local_best_param,i,5):
                       # print 'large weights'
                        #omit negative noces
                        #split zero nodes
                        (p.param,zero_nodes)=OmitNegativeNodes(EnergyFunction,p.local_best_param)
                        p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param,limit=float('inf'))
##                        print 'error before splitting nodes',error
##                        print 'params before',p.param
                        p.param = SplitNodes(EnergyFunction,p.local_best_param,zero_nodes)
##                        p.local_best_error =  EnergyFunction.truncated_functions(p.local_best_param,limit=float('inf'))
##                        print 'error after splitting nodes',error
##                        print 'params after',p.param 
                        
            if p.local_best_error < BestError:
                BestError = p.local_best_error
                BestParticle = i
        for p in particle_list:
            lambda1 = numpy.array(map(lambda x:1-2*random.random(), p.param))
            lambda2 = numpy.array(map(lambda x:1-2*random.random(), p.param))
            Best = particle_list[BestParticle].local_best_param
            p.velocity = c1*lambda1*(p.local_best_param-p.param) + c2*lambda2*(Best-p.param)
            #bound(p.velocity,[100,5,5,5])
            p.param = p.param +p.velocity + momentum*p.velocity_old
            p.velocity_old = copy.deepcopy(p.velocity)
            momentum = momentum*decay
        particle_indices = []
        for i in range(k):
            particle_indices+=[[]]

        for i,p in enumerate(particle_list):
            p.closest_cluster = -1
            p.closest_dist = float('inf')
            for j,c in enumerate(cluster_centers):
                dist=((p.param-c.param)**2).sum()
                if dist < p.closest_dist:
                    p.closest_dist = dist
                    p.cluster = j
            print particle_indices
            print "cluster index ",p.cluster
            particle_indices[p.cluster]+=[i]
        #average_particles = copy.deepcopy(cluster_centers)    
        for j,c in enumerate(cluster_centers):
            average_particle=c.param
            if len(particle_indices[j]) > 0:
                average_particle=0*average_particle
                for i in range(0,len(particle_indices[j])):
                    average_particle+=particle_list[i].param
                average_particle=average_particle/len(particle_list[i].param)
            cluster_centers[j].param = average_particle
            cluster_centers[j].error = EnergyFunction.truncated_functions(cluster_centers[j],limit=float('inf'))
        cluster_centers.sort(key=lambda x:x.error)
        #determine error for each cluster_center and sort
        cluster_centers=NelderMeade(EnergyFunction, cluster_centers, MaxIterations=20, ErrorThreshold=1e-5)
        cluster_best = cluster_centers[0]
        print "particle best error ", particle_list[BestParticle].local_best_error," cluster_best.error", cluster_best.error
        
        it+=1
    #sort particles in order
    best_solution = particle_list[BestParticle]
    if cluster_best.error < best_solution.local_best_error:
        best_solution =cluster_best
    print "best error ", best_solution.error
    return best_solution


#assumes the vertexList is in order??
def NelderMeade(EnergyFunction, vertexList,MaxIterations=30,ErrorThreshold=1e-4):
    reflection=1
    expansion=2
    contraction=0.5
    reduction=0.5
    n = len(vertexList)-2
    m = len(vertexList[0].param)
    Centroid=numpy.array([0]*m)
    reflection_num=0
    contraction_num=0
    expansion_num=0
    reduction_num=0

    iteration=0
    while iteration < MaxIterations and vertexList[0].error > ErrorThreshold:
        print "iteration ", iteration, " error: ", vertexList[0].error
        Centroid=0*Centroid
        for i in range(n):
            v = vertexList[i]
            Centroid+=v.param
        Centroid=Centroid/n
        reflected = Centroid + reflection*(Centroid - vertexList[n+1].param)
        reflected_error = EnergyFunction.functions(reflected)
        if vertexList[0].error <= reflected_error \
           and reflected_error < vertexList[n].error:
            reflection_num+=1
            vertexList[n+1].param = reflected
            vertexList[n+1].error = reflected_error
        if reflected_error <= vertexList[0].error:
            expansion = Centroid + expansion*(Centroid - vertexList[n+1].param)
            expansion_error = EnergyFunction.functions(reflected)
            if expansion_error < reflected_error:
                expansion_num+=1
                vertexList[n+1].param = expansion
                vertexList[n+1].error = expansion_error
            else:
                reflection_num+=1
                vertexList[n+1].param = reflected
                vertexList[n+1].error = reflected_error
        if reflected_error >= vertexList[n].error:
            contraction_num+=1
            contraction = vertexList[n+1].param+contraction*(Centroid - vertexList[n+1].param)
            #print contraction
            contraction_error = EnergyFunction.functions(contraction)
            if contraction_error < vertexList[n+1].error:
                vertexList[n+1].param = contraction
                vertexList[n+1].error = contraction_error
            else:
                reduction_num+=1
                for i in range(1,n+1):
                    vertexList[i].param = vertexList[0].param +reduction*\
                                          (vertexList[0].param-vertexList[i].param)
                    vertexList[i].error = EnergyFunction.functions(vertexList[i].param)
        vertexList.sort(key=lambda x:x.error)
        iteration+=1
    return vertexList
        
def updateError(error, param, best_error, best_param):
    print "updating error ", error, " ", best_error
    if error <best_error:
        print "successs"
        best_param = copy.deepcopy(param)
        best_error = error
    return (best_param,best_error)

def mutateParameters(param_list, ranges, radius):
    
    for p in param_list:
        lamb = numpy.array(map(lambda x:1-2*random.random(), p.param))
        ranges_vals= numpy.array(ranges*(len(p.param)/4))
        param_list +=ranges_vals*lamb*radius
class Sol:
    def __init__(self,param,error):
        self.error = error
        self.param = param

##Optimisation is considered to be an object that contains an optimise method

class Optimisation:

    ##optmise takes the Energy Function, the initial parameters, and the number of
    ##runs of the sequences and whether to employ global search methods like particle
    ##swarm optimisation

    def optimise4(EnergyFunction, param, runs=3,global_search_opt=True):
##        print "param", param
        initial_error = EnergyFunction.functions(0*param)
        best_error = initial_error
        best_param = copy.copy(param)
        
        errorThreshold = 0.01*initial_error
        print "errorThreshold: " +  str(errorThreshold)
        centre = copy.deepcopy(param)
        radius = numpy.array([1.0,20.0,20.0,20.0]*(len(param)/4))
        numSolutions=20
        overall_best_error = initial_error
        overall_best_param = copy.copy(param)
        for i in range(runs):
            print "commencing Monte Carlo: "
            solution_list = MonteCarlo(EnergyFunction, centre, radius, numSolutions)
            if global_search_opt:
                (best_param,best_error) = updateError(solution_list[0].error, solution_list[0].param, best_error, best_param)
##                print solution_list[0].param
##                jacobian = EnergyFunction.jacobians_residual(solution_list[0].param)
##                print jacobian
                #print map(lambda x:x.param, solution_list)
                print "commencing Particle Swarm Optimisation"
                solution_list =ParticleSwarmOptimisation(EnergyFunction, solution_list,
                                      MaxIterations=100, ErrorThreshold=errorThreshold)
        ##        proximalInitialSolutions
                (best_param,best_error) = updateError(solution_list[0].error, solution_list[0].param, best_error, best_param)

                #mutateParameters(solution_list, [5.0,2.0,0.5,0.5], 1)
        ##        param = fmin(EnergyFunction.functions,solution_list[0].param)
        ##        solution_list =NelderMeade(EnergyFunction, solution_list)
        ##        (best_param,best_error) = updateError(solution_list[0].error, solution_list[0].param, best_error, best_param)
                param = solution_list[0].param
    ##        (best_param,best_error) = updateError(solution_list[0].error, solution_list[0].param, best_error, best_param)
    ##        param1 = fmin_bfgs(EnergyFunction.functions, param,
    ##                          fprime=EnergyFunction.jacobians,
    ##                          maxiter=10,errorThreshold=errorThreshold)
##            param = fmin_bfgs(EnergyFunction.functions, param,
##                              maxiter=100,errorThreshold=errorThreshold)
            param = solution_list[0].param
            old_error = solution_list[0].error
##            print "param", param, sum(param)
            print "old error ",old_error
            print "commencing the Levenberg-Marquardt method" 
           ## param = leastsq(EnergyFunction.functions_diff,param,Dfun = EnergyFunction.jacobians)[0]
            param = leastsq(EnergyFunction.functions_diff,param,col_deriv=True)[0]
    ##        param2 = fmin_cg(EnergyFunction.functions, param,
    ##                          maxiter=100)

            
            new_error = EnergyFunction.functions(param)
            error = new_error
            print "new error ",new_error
##            print "param", param, "sum", sum(param)
            print "commencing quasi-Newton's BFGS method" 
            param = fmin_bfgs(EnergyFunction.functions, param,
                      fprime=EnergyFunction.jacobians,
                      maxiter=10)
            new_error = EnergyFunction.functions(param)
            error = new_error
##            print "new error2 ",new_error 
##            success_measure = math.log(old_error)/math.log(10) - math.log(new_error)/math.log(10)
##            print "success_measure: ",success_measure
            if errorThreshold < error:
                print "commencing gradient descent" 
                param = fmin_bfgs(EnergyFunction.functions, param,maxiter=10)
                error = EnergyFunction.functions(param)
            (best_param,best_error) = updateError(error, param, best_error, best_param)
            (overall_best_param,overall_best_error) =updateError(best_error, best_param, overall_best_error, overall_best_param)
            centre = best_param
            radius = 0.5*centre
        return overall_best_param
    optimise4 = Callable(optimise4)

 
    
