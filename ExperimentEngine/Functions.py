from sympy import *
import numpy
from numpy import matrix
import copy

#Generate static calculation code required in preprocessing#
def differentiate(fn, x=Symbol('x'),xn=0,y=Symbol('y'), yn=0):
    d_fn = copy.deepcopy(fn)
    for i in range(xn):
        d_fn = diff(d_fn, x)
    for i in range(yn):
        d_fn = diff(d_fn, y)
    return d_fn


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
class Functions:
    def __init__(self):
        self.functions={}
    
    def Euclidean(self):
        k=4
        x = Symbol('x')
        y = Symbol('y')
        a = Symbol('x0')
        b = Symbol('y0')
        d = Symbol('d')
        fn = sqrt((x-a)**2 + (y-b)**2 +d**2)
        for i in range(5):
            j=0
            while j <= k:
                euc_str = str(differentiate(fn, xn=i, yn=j))
                euc_str =euc_str.replace("/2","/2.0")
                self.functions["euclidean["+str(i)+"]["+str(j)+"]"] = euc_str
               # print "elif xn== "+ str(i) +  " and yn== " + str(j) + ":"
               # print "    E="+self.functions["euclidean["+str(i)+"]["+str(j)+"]"]
                j+=1
            k-=1
            
    def TPS(self):
        r =Function('r(x, y)')
        x = Symbol('x')
        y = Symbol('y')
        fn = r(x,y)**2*log(r(x,y)**2)
        k=4
        for i in range(5):
            j=0
            while j <= k:            
                #self.functions={ "tps_"+str(i)+"_"+str(j): differentiate(fn, xn=i, yn=j)}
                #print i,j, self.functions["tps_"+str(i)+"_"+str(j)]

                d_str = str(differentiate(fn, xn=i, yn=j))
                d_str = d_str.replace("(x, y)","")
                l=d_str.find("D", 0)
                #print d_str
                while l != -1:       
                    start_l = l
                    l+=2
                    end_l = matchBracket(d_str, l, '(')
                    x_count = 0
                    y_count = 0
                    term = d_str[start_l:end_l]
                    p=0
                    while p !=-1:
                        p=term.find("x", p)

                        if p !=-1:
                            x_count+=1
                            p+=1
                    p=0
                    while p !=-1:
                        p=term.find("y", p)
                        if p !=-1:
                            y_count+=1
                            p+=1   
                    #fn_str = "functions[euclidean_"+str(x_count) +"_" +str(y_count)+"]"
                    #fn_str = "eval(func[\"euclidean_"+str(x_count) +"_" +str(y_count)+"\"])"
                    fn_str = "R["+str(x_count) +"][" +str(y_count)+"]"      
                    d_str= d_str[0:start_l] + fn_str+ d_str[end_l:len(d_str)]
                    l = d_str.find("D", 0)
                d_str = d_str.replace("r","R[0][0]")
                d_str = d_str.replace("R","r")
                d_str = d_str.replace("log","math.log")
                self.functions["TPS_"+ str(i)+"_"+str(j)]=d_str
               # print "elif xn== "+ str(i) +  " and yn== " + str(j) + ":"
               # print "T="+self.functions["TPS_"+ str(i)+"_"+str(j)]
                #print d_str
                j+=1
            k-=1

    #length factors here
    def LengthFactors(self):
        #self.functions={}
        k=4
        x = Symbol('x')
        y = Symbol('y')
        X = Function('X(x, y)')
        Y = Function('Y(x, y)')
        r = Symbol('r')
        fn = r - X(x, y)**2 - Y(x, y)**2
        for i in range(5):
            j=0
            while j <= k:
                lf_str = str(differentiate(fn, xn=i, yn=j))
                lf_str = lf_str.replace("(x, y)","")
                l=lf_str.find("D", 0)
                while l != -1:
                    start_l = l
                    l+=2
                    end_l = matchBracket(lf_str, l, '(')
                    x_count = 0
                    y_count = 0
                    term = lf_str[start_l:end_l]
                    if term.find("X",0)>0:
                        functionType="X"
                    else:
                        functionType="Y"
                    p=0             
                    while p !=-1:
                        p=term.find("x", p)

                        if p !=-1:
                            x_count+=1
                            p+=1
                    p=0
                    while p !=-1:
                        p=term.find("y", p)
                        if p !=-1:
                            y_count+=1
                            p+=1   
                    #fn_str = "functions[euclidean_"+str(x_count) +"_" +str(y_count)+"]"
                    fn_str = functionType+"["+str(x_count) +"][" +str(y_count)+"]"
                    lf_str= lf_str[0:start_l] + fn_str+ lf_str[end_l:len(lf_str)]
                    l = lf_str.find("D", 0)
                l=lf_str.find("X", 0)
                while l != -1 and l < len(lf_str):
                    if l==len(lf_str)-1 or lf_str[l+1]!='[':
                        lf_str= lf_str[0:l+1] + '[0][0]'+ lf_str[l+1:len(lf_str)]
                    l=lf_str.find("X", l+1)
                l=lf_str.find("Y", 0)
                while l != -1 and l < len(lf_str):
                    if l==len(lf_str)-1 or lf_str[l+1]!='[':
                        lf_str= lf_str[0:l+1] + '[0][0]'+ lf_str[l+1:len(lf_str)]
                    l=lf_str.find("Y", l+1)
                    
                self.functions["length_factors_"+str(i)+"_"+str(j)] = lf_str
               # print "elif xn=="+ str(i) +  " and yn==" + str(j) + ":"
               # print "lf="+self.functions["length_factors_"+str(i)+"_"+str(j)]
                j+=1
            k-=1
        self.functions["length_factors_"+str(i)+"_"+str(j)] = lf_str

    def Interpolation(self):
        k=3
        x0 = Symbol('x0')
        y0 = Symbol('y0')
        x = Symbol('x')
        y = Symbol('y')
        v = Symbol('v')
        fn = v/sqrt((x-x0)**2+(y-y0)**2)
        for i in range(4):
            j=0
            while j <= k:
                lf_str = str(differentiate(fn, xn=i, yn=j))
                self.functions["interpolant_"+str(i)+"_"+str(j)] = lf_str
               # print i,j, self.functions["interpolant_"+str(i)+"_"+str(j)]
                j+=1
            k-=1
        self.functions["interpolant_"+str(i)+"_"+str(j)] = lf_str   
    
##functions = Functions()
##functions.Euclidean()
##functions.TPS()


##functions.LengthFactors()
##functions.Interpolation()
##func = functions.functions
###print functions.functions["euclidean_0_0"]
##print "hi again"
