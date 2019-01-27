from numpy import matrix
import numpy as np
from sympy import *
import copy
#from copy import *

def differentiate(fn, x=Symbol('x'),xn=0,y=Symbol('y'), yn=0):
    d_fn = copy.deepcopy(fn)
    #print d_fn, xn, yn, diff(Lm)
    for i in range(yn):
        d_fn = diff(d_fn, y)
    for i in range(xn):
        d_fn = diff(d_fn, x)

##    d_fn = together(diff(d_fn, x,xn))
##    d_fn = together(diff(d_fn, y,yn))
        
    return d_fn

def dot(a,b):
    return a[0]*b[0] + a[1]*b[1]

def grad(a):
    return np.array([diff(a,x), diff(a,y)])
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
def includeBrackets(d_str, f_str):
    l=0
    l = d_str.find(f_str, l)
    while l != -1:   
        #print l
        brack_pos = l + len(f_str)
       # print brack_pos
        if brack_pos == len(d_str) or d_str[brack_pos]!='[':
            #print brack_pos == len(d_str) or d_str[brack_pos]!='['
            d_str= d_str[0:brack_pos]+ '[0][0]' + d_str[brack_pos:len(d_str)] 
            #print d_str,l
        else:
            l=brack_pos
        l = d_str.find(f_str, l)
    return d_str 


def removeDerivativeString(d_str):
        l=d_str.find("D", 0)

        while l != -1:       
            start_l = l
            comma_l = d_str.find(",", start_l)
            fn_str = d_str[start_l+2:comma_l]
            l+=2
            end_l = matchBracket(d_str, l, '(')
            x_count = 0
            y_count = 0
            term = d_str[start_l:end_l]
            p=comma_l-start_l
            while p !=-1:
                p=term.find("x", p)

                if p !=-1:
                    x_count+=1
                    p+=1
            p=comma_l-start_l
            while p !=-1:
                p=term.find("y", p)
                if p !=-1:
                    y_count+=1
                    p+=1
      
            #fn_str = "functions[euclidean_"+str(x_count) +"_" +str(y_count)+"]"
            #fn_str = "eval(func[\"euclidean_"+str(x_count) +"_" +str(y_count)+"\"])"
           # print fn_str
            fn_str += "[" + str(x_count) + "][" + str(y_count) + "]"   
            d_str= d_str[0:start_l] + fn_str+ d_str[end_l:len(d_str)]
            l = d_str.find("D", 0)
        return d_str
x=Symbol('x')
y=Symbol('y')
p=Symbol('p')
Ld = Function('Ld(x, y)')
Lm = Function('Lm(x, y)')
Ad = Function('Ad(x, y)')
n = Function('n(x, y)')
Ldy = Function('Ldy(x, y)')
Ldx = Function('Ldx(x, y)')
Ldy = Function('Ld(x, y)')
ny = Function('ny(x, y)')
nx = Function('nx(x, y)')
Ady = Function('Ady(x, y)')
Ady = Function('Adx(x, y)')
g = Function('g(x, y)')
n = Function('n(x, y)')
Nx = Function('Nx(x, y)')
Adx = Function('Adx(x, y)')
Ny = Function('Ny(x, y)')
N = Function('N(x, y, p)')
normal = np.array([nx(x,y), ny(x,y)])

K = Symbol('K')
a = Symbol('a')
fudge = K*(1 - exp(-(a*Lm(x, y))))
##num0=Ld(x, y)*Lm(x, y)*(g(x, y) - ((Adx(x, y)*nx(x, y) +Ady(x, y)*ny(x, y)) + N(x, y)*(Ldx(x, y)*nx(x, y) +  Ldy(x, y)*ny(x, y)) + Ld(x, y)*(Nx(x, y)*nx(x, y)+Ny(x, y)*ny(x, y))))
##S0=str(num0).replace('(x, y)',"")
#print S0

##num1 = Ld(x, y)*Lm(x, y)*(g(x, y) - dot(grad(Ad(x, y)) + grad(N(x, y, p)*Ld(x, y)), normal))
#num1 = Ld(x, y)*Lm(x, y)*dot(grad(Ad(x, y)) + grad(N(x, y)*Ld(x, y)), normal)
##dom1 = Ld(x, y)*dot(grad(Lm(x, y)),normal) + fudge
#sol = (dot(grad(Ad(x, y)) + grad(N(x, y)*Ld(x, y)), normal))/(Ld(x, y)*dot(grad(Lm(x, y)),normal) + fudge)
##k=3
####
####
##for i in range(4):
##    j=0
##    while j <= k:
##        print i,j
##        s = num1/dom1
##        ds_ij = differentiate(s,xn=i,yn=j)
##        #ds_ij = simplify(diff(ds_ij, p))
##        S=str(ds_ij).replace('(x, y)',"")
##        S=str(S).replace('(x, y, p)',"")
##        #ds_ij = collect(ds_ij,[Ld(x, y),N(x,y),Lm(x,y),Ad(x,y),nx(x,y), ny(x,y),g(x,y)])
##        
##        #S=str(ds_ij).replace('(x, y)',"")
##        S=removeDerivativeString(S)
##        S=includeBrackets(S, 'Ld')
##        S=includeBrackets(S, 'Lm')
##        S=includeBrackets(S, 'N')
##        S=includeBrackets(S, 'Ad')
##        S=includeBrackets(S, 'nx')
##        S=includeBrackets(S, 'ny')
##        S=includeBrackets(S, 'g')
##        print S
##        print "count " + str(S.count('[')/2)
##        j+=1
##    k-=1
num1 = (g(x, y) - dot(grad(Ad(x, y)) + grad(N(x, y, p)*Ld(x, y)), normal))
###num1 = Ld(x, y)*Lm(x, y)*dot(grad(Ad(x, y)) + grad(N(x, y)*Ld(x, y)), normal)
##dom1 = Ld(x, y)*dot(grad(Lm(x, y)),normal) + fudge
#numerator
k=3
for i in range(4):
    j=0
    while j <= k:
        #print i,j
        s = num1
        ds_ij = differentiate(num1,xn=i,yn=j)
        ds_ij = diff(ds_ij, p)
       # ds_ij = collect(ds_ij,[Ld(x, y),N(x,y),Lm(x,y),Ad(x,y),nx(x,y), ny(x,y),g(x,y)]
        S=str(ds_ij).replace('(x, y)',"")
        S=str(S).replace('(x, y, p)',"")
        
        
        #S=str(ds_ij).replace('(x, y)',"")
        S=removeDerivativeString(S)
        S=includeBrackets(S, 'Ld')
        S=includeBrackets(S, 'Lm')
        S=includeBrackets(S, 'N')
        S=includeBrackets(S, 'Ad')
        S=includeBrackets(S, 'nx')
        S=includeBrackets(S, 'ny')
        S=includeBrackets(S, 'g')
        
        print "if xn == "+ str(i) + " and yn == " + str(j)
        print "    n = " +S
        #print "count " + str(S.count('[')/2)
        j+=1
    k-=1

#denominator
##k=3
##for i in range(4):
##    j=0
##    while j <= k:
##       # print i,j
##        s = dom1
##        ds_ij = differentiate(s,xn=i,yn=j)
##        #ds_ij = simplify(diff(ds_ij, p))
##       # ds_ij = collect(ds_ij,[Ld(x, y),N(x,y),Lm(x,y),Ad(x,y),nx(x,y), ny(x,y),g(x,y)]
##        S=str(ds_ij).replace('(x, y)',"")
##        S=str(S).replace('(x, y, p)',"")
##        
##        
##        #S=str(ds_ij).replace('(x, y)',"")
##        S=removeDerivativeString(S)
##        S=removeDerivativeString(S)
##        S=includeBrackets(S, 'Ld')
##        S=includeBrackets(S, 'Lm')
##        S=includeBrackets(S, 'N')
##        S=includeBrackets(S, 'Ad')
##        S=includeBrackets(S, 'nx')
##        S=includeBrackets(S, 'ny')
##        S=includeBrackets(S, 'g')
##
##        print "if xn == "+ str(i) + " and yn == " + str(j)
##        print S
##        #print "count " + str(S.count('[')/2)
##        j+=1
##    k-=1

#quotient
##k=3
##num =Function('num(x, y)')
##dom =Function('dom(x, y)')
##f = num(x, y)/dom(x, y)
##for i in range(4):
##    j=0
##    while j <= k:
##        #print i,j
##        s = num1/dom1
##        ds_ij = differentiate(f,xn=i,yn=j)
##        #ds_ij = simplify(diff(ds_ij, p))
##        S=str(ds_ij).replace('(x, y)',"")
##        S=str(S).replace('(x, y, p)',"")
##        #ds_ij = collect(ds_ij,[Ld(x, y),N(x,y),Lm(x,y),Ad(x,y),nx(x,y), ny(x,y),g(x,y)])
##        
##        #S=str(ds_ij).replace('(x, y)',"")
##        S=removeDerivativeString(S)
##        S=includeBrackets(S, 'dom')
##        S=includeBrackets(S, 'num')
## 
##        print "if xn == "+ str(i) + " and yn == " + str(j)
##        print "    q = " +S
##        #print "count " + str(S.count('[')/2)
##        j+=1
##    k-=1

#triplet
##k=3
##S =Function('S(x, y)')
##f =Lm(x, y)*Ld(x, y)*S(x, y)
##for i in range(4):
##    j=0
##    while j <= k:
##       # print i,j
##        s = num1/dom1
##        ds_ij = differentiate(f,xn=i,yn=j)
##        #ds_ij = simplify(diff(ds_ij, p))
##        S=str(ds_ij).replace('(x, y)',"")
##        S=str(S).replace('(x, y, p)',"")
##        #ds_ij = collect(ds_ij,[Ld(x, y),N(x,y),Lm(x,y),Ad(x,y),nx(x,y), ny(x,y),g(x,y)])
##        
##        #S=str(ds_ij).replace('(x, y)',"")
##        S=removeDerivativeString(S)
##        S=includeBrackets(S, 'S')
##        S=includeBrackets(S, 'Ld')
##        S=includeBrackets(S, 'Lm')
##        print "if xn == "+ str(i) + " and yn == " + str(j)
##        print "    s = " + S
##       # print "count " + str(S.count('[')/2)
##        j+=1
##    k-=1

##        self.representations
##        
##        self.sol_rep_list         
##        self.sol_rep_info_list 
##        self.sol_mon_prob_list 
##        self.sol_mon_prob_info_list
##        self.sol_prob_mon_DE_list
##        self.sol_prob_mon_source_list
##    def __init__(self, prob_info, basis_function = "sigmoid", representation_type = "implicit",name="",prob=ProblemInformation()):
##        self.name = name
##        self.basis_function = basis_function
##        self.representation_type = representation_type
##        self.prob=prob
##    def setMcfall(self, mcfall):
##        self.mcfall =mcfall
##
##
##            def __init__(self, probs=ProblemInformationList(),name="",basis_function = "sigmoid", representation_type = "implicit",\
##                 prob_type = "de",param=[0,0,0,0],\
##                 implicit_include_BC=True):
##
