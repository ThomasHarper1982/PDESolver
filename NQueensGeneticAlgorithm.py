from tkinter import *
import tkinter as tk
import random
import numpy
import operator

'''
Optimisation algorithm and animation:

The N Queens problem and applying Genetic Algorithms and Taboo search to solve it.
Using tkinter, the program displays an N by N chessboard with N queens, showing the current best
solution of N queens problem. The goal is to find a state where no queen can attack.
This is equivalent to an objective function value of 0.
The python shell shows the current iteration and objective function value.
N is a hardcoded parameter.
Genetic Algorithms applies powerful concepts from evolutionary
biology including crossover and mutation of genomes in reproduction. This method is conducive
to solving discrete problems where solutions can be represented as a vector.
Taboo is a heuristic for avoiding areas called local minima
which have already been explored and do not yield better solutions by exploring neighbouring solutions.
Requires Python 3.4, numpy and tkinter.
@author Thomas Harper
'''

class Solution:

    def __init__(self, vector =[], objValue=float('inf')):
        '''Data Structure for each solution, consists of sequence of queens
    and objective function'''
        self.vector = vector; ## array of ints
        self.objValue = objValue; ##objective function value -- the number of attacking positions
                                  ##
                                                    
        
class NQueensGeneticAlgorithm(tk.Frame):

    def key(self,solution):
        '''Approximately unique hashing function for each solution'''
        return sum(map(operator.mul, list(map(pow,[15] * len(solution.vector), range(len(solution.vector)))),solution.vector));

    def dirac_delta(self,x):
        '''Function returns 1 when input x = 0, else 0'''
        y = 0;
        if (x==0):
            y = 1;
        return y;


    
    def objectiveFunction(self, solution):
        ''' Objective Function for the N queens problem'''
        keyV = self.key(solution);
        if (self.solutions.__contains__(keyV)):
            return self.solutions[keyV];
        obj = 0;
        n = len(solution.vector);
        Q = solution.vector;
        for q in range(1,n+1):
            for p in range(1,n+1):
               if (p != q):
                    k1 = -min(q, Q[q-1])+1;
                    k2 = min(n-q, n-Q[q-1]);
                    k3 = -min(q, n-Q[q-1])+1;
                    k4 = min(q-1,Q[q-1]);
                    obj = obj + self.dirac_delta(p-q) + self.dirac_delta(Q[p-1]-Q[q-1]);   #row/column attacks
                    for k in range(k1,k2+1): #diagonal attacks
                     obj = obj+ self.dirac_delta(q + k - p)*self.dirac_delta(Q[q-1]+k-Q[p-1]);

                    for k in range(k3,k4+1): #diagonal attacks
                     obj = obj + self.dirac_delta(q + k - p)*self.dirac_delta(Q[q-1]-k-Q[p-1]);
        self.solutions[keyV] = obj;
        return obj;
        
    def generateIndividual(self, numQueens):
        '''Generate a random solution'''
        S = [];
        for i in range(numQueens):
            S.append(random.randint(0,numQueens-1))
        return S;


    def generateInitialPopulation(self):
        '''Generate a random initial population'''
        pop = [];
        for i in range(self.population):
            p = Solution();
            p.vector = self.generateIndividual(self.numQueens)
            p.objValue = self.objectiveFunction(p)
            pop.append(p);

        return pop;


    def reviseObjectiveFunctionsTaboo(self):
        '''For all members of the population, if any have been visited x times
    treat it as a local minima and set objective function value very high.
    This is how we implement taboo policy.'''
        x = 10;
        for i,p in enumerate(self.pop):
            keyV = self.key(p);
            if self.taboo.__contains__(keyV) and self.taboo[keyV]>x:
                p.objValue=1000;
                self.solutions[keyV] = 1000;

    def chooseGoodIndividual(self, exclude=-1):
        '''Choose a strong individual based on a probability distribution that
    is based on all population members'''
        
        #will break by divide by 0 if obj=0, i.e. optimal solution is found
        points = numpy.array([0.0]*self.population);
        for i,p in enumerate(self.pop):
            if (p.objValue==0):
                index = i;
                return index;
            points[i] = 1/p.objValue;
        points = points/sum(points);
            
        dist = numpy.cumsum(points);
        r = random.random();
        index = 0;

        for i,e in enumerate(dist):
            if exclude != i and e > r:
                 index = i;
                 break;

        return index;
    

    def eightQueensReproduce(self, x, y):
        '''With 2 parents produce a child using crossover and mutation.
    Then generate child's objective function value'''   
        L = self.numQueens;
        r1 = random.randint(1,L-1);

        xvector = x.vector;
        yvector = y.vector;            
    
        c1_vector = xvector[0:r1] + yvector[r1:L]; 
        c2_vector = yvector[0:r1]+ xvector[r1:L]; 

        r = random.random();
        c_vector = []
        if (r<0.5):
            c_vector = c1_vector;
        else:
            c_vector = c2_vector; 

        
        for i in range(0,L):
            if (random.random() < self.maxMutationRate):
                c_vector[i] = random.randint(0,L-1);

        child = Solution();
        child.vector = c_vector;

        child.objValue = self.objectiveFunction(child);
        return child;
     

    def geneticAlgorithmMainLoop(self):
        '''Main loop for the genetic algorithm.'''
        change = True;
        if self.t < self.maxIterations and self.overallBest.objValue!=0: 
            print('iteration :' + str(self.t))
            if (self.t%10==1):
                self.reviseObjectiveFunctionsTaboo();

            self.sortPop();
            topPop = int(self.retentionRate*self.population);

            for k in range(topPop+1, self.population):
                xi = self.chooseGoodIndividual();
                yi = self.chooseGoodIndividual(exclude=xi);
               
                x = self.pop[xi];
                y = self.pop[yi];
                c = self.eightQueensReproduce(x, y);
                self.pop[k] = c;
            #determine if redraw is needed
            change = self.currentBest != self.pop[0];
                    
            self.currentBest = self.pop[0];


                
            if self.overallBest.objValue > self.currentBest.objValue:
                self.overallBest = self.currentBest;

            keyV = self.key(self.currentBest);
            if self.taboo.__contains__(keyV):
                self.taboo[keyV] = self.taboo[keyV] + 1;
            else:
                self.taboo[keyV] = 0;
                
            self.historyBestObj[self.t] = self.overallBest.objValue ;
            self.historyCurrentBestObj[self.t] = self.currentBest.objValue;
            
            print ('objective function :' + str(self.currentBest.objValue))
          #  print ('overall best :' + str(self.overallBest.objValue))
            self.t+=1;

            self.drawBoard();
            self.parent.after(10, self.geneticAlgorithmMainLoop);
        else:
            
            print('solution : ' , self.overallBest.vector)


    def placeCurrentBest(self):
        '''draws each queen in a solution'''
        vector = self.currentBest.vector;
        
        for col,row in enumerate(vector):
            self.drawQueen(ax=2, bx=1 + col*self.size,
                           ay=2, by=5 + row*self.size)

    def sortPop(self):
        self.pop.sort(key=lambda x: x.objValue)

    def drawQueen(self,ax=2,bx=3, ay=2,by=5):
        X = [ax*1+bx, ax*3+bx, ax*5+bx, ax*7+bx,
             ax*9+bx, ax*11+bx, ax*13+bx, ax*13+bx, ax*1+bx];
        Y = [ay*0+by, ay*3+by, ay*0+by, ay*3+by,
             ay*0+by, ay*3+by, ay*0+by, ay*9+by, ay*9+by];
        self.canvas.create_line(X[0],Y[0],X[1],Y[1], width=3);
        self.canvas.create_line(X[1],Y[1],X[2],Y[2], width=3);
        self.canvas.create_line(X[2],Y[2],X[3],Y[3], width=3);
        self.canvas.create_line(X[3],Y[3],X[4],Y[4], width=3);
        self.canvas.create_line(X[4],Y[4],X[5],Y[5], width=3);
        self.canvas.create_line(X[5],Y[5],X[6],Y[6], width=3);
        self.canvas.create_line(X[6],Y[6],X[7],Y[7], width=2);
        self.canvas.create_line(X[7],Y[7],X[8],Y[8], width=3);
        self.canvas.create_line(X[8],Y[8],X[0],Y[0], width=2);

    

    def drawBoard(self):
        '''Redraw the board, possibly in response to window being resized'''
        self.canvas.delete("all")
        self.size = min(self.xsize, self.ysize)
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
##            if self.numQueens%2==0:
##                color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
        
                #if self.numQueens%2==1:
                if (row+col)%2==0:
                    color = self.color1
                else:
                    color = self.color2

                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2
        
        self.placeCurrentBest();
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")  

    def refresh(self, event):
        self.xsize = int((event.width-1) / self.columns)
        self.ysize = int((event.height-1) / self.rows)
        self.drawBoard();

    def __init__(self, parent, numQueens):

  #      random.seed(1) #setting the seed is helpful for testing and debugging
        self.parent = parent; #root tk object
        
        self.numQueens = numQueens;
        self.population = 50;
        self.maxIterations = 3000;
        self.retentionRate = 0.1; #this is the ratio of the best members that are retained every iteration
        self.maxMutationRate = 0.05; 

        self.solutions = dict(); #this is a dictionary (like a cache) of all objective function values mapped by solution vector key
        self.taboo = dict(); #this is a dictionary of how many times a solution has been visited, also mapped by solution vector key
        self.pop = self.generateInitialPopulation();

        self.sortPop();
        self.overallBest = self.pop[0];
        self.currentBest = self.pop[0];
        self.historyBestObj = [0]*self.maxIterations;
        self.historyCurrentBestObj = [0]*self.maxIterations;       

        
        self.rows = self.numQueens
        self.columns = self.numQueens
        self.size = 32
        self.color1 = 'white'
        self.color2 = 'blue'
        self.pieces = {}
        self.t = 0;
        canvas_width = self.columns * self.size
        canvas_height = self.rows * self.size

        self.xsize = canvas_width;
        self.ysize = canvas_height;
        

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height, background="bisque")
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh);


    
    
if __name__ == '__main__':
    root = tk.Tk()
    numQueens = 18 ##feel free to change this parameter >= 4
    view = NQueensGeneticAlgorithm(root, numQueens);
    view.pack(side="top", fill="both", expand="true", padx=4, pady=4)
    root.wm_title("Solving " +  str(numQueens) + " Queens with Genetic Algorithm and Taboo")
    
   # view.addpiece("player1", 1,0)
    root.after(0, view.geneticAlgorithmMainLoop);
   # print('entering mainloop')
    root.mainloop();
    
