##The main file for the Neural Network based Differential Equation Experimentation
##Engine, F5 to run
import os, os.path, sys, Tix
from Tkconstants import *
from ProblemInformation import *
from Solutions import *
from McFall import *
from EnergyFunction import *
import sympy
from sympy import *
from Optimisation import *
from InterpolationMethod import *
import tkMessageBox, traceback
import random
import numpy

TCL_DONT_WAIT       = 1<<1
TCL_WINDOW_EVENTS   = 1<<2
TCL_FILE_EVENTS     = 1<<3
TCL_TIMER_EVENTS    = 1<<4
TCL_IDLE_EVENTS     = 1<<5
TCL_ALL_EVENTS      = 0

#Solution Representation and Problem information combined into a new object 
class RepresentationProblem:
    def __init__(self, probs,name="",basis_function = "sigmoid", representation_type = "implicit",\
                 prob_type = "de",param=np.array([0,0,0,0]), implicit_include_BC=True):
        self.name = name
        self.basis_function = basis_function
        self.representation_type = representation_type
        self.probs=probs
        self.prob= probs.prob_list[0]
        self.prob_type = prob_type
        self.param=param
        self.num_nodes = param/4
        self.implicit_include_BC=implicit_include_BC
        self.error = inf
##      
    def setMcfall(self, mcfall):
        self.mcfall =mcfall
        self.representation_type = "explicit"

##    def viewErrorFunction(self):
##        N=[0,0,0,0]
##        #for each problem
##        for t in self.probs.prob_list:
##            EnergyFunction.function
        
class Test:

    def __init__(self, top):
        self.root = top
        self.exit = -1

##        self.dir = None             # script directory
        self.balloon = None         # balloon widget
        self.useBalloons = Tix.StringVar()
        self.useBalloons.set('0')
##        self.statusbar = None           # status bar widget
##        self.welmsg = None          # Msg widget
##        self.welfont = ''           # font name
##        self.welsize = ''           # font size
        self.representations_list=[]


        #we have an empty list of problems
        self.problems=[]
        self.representations=[]
        self.problem_list = []
        self.DE_list = []
        self.source_func = []
        self.point1_list = []
        self.point2_list = []
        self.view_mode_opt = "source"
        self.func_assign_opt = "solution"
        self.explicit_viewing_opt="implicit"
        self.basis_function_opt = 'sigmoid'
        self.bound_representation_opt ='implicit'
        self.view_solution_prob_val = "source"
        self.prob_assign_opt="de"
        self.implict_BC_option=False
        self.FirstTerm_option= True
        self.sample_independent= False
        self.global_search_opt = False
        self.sol_current_prob_name=""
        self.sol_current_rep_name=""
        self.sol_defined=0
        self.num_nodes=10
        self.num_samples=1
        self.num_runs=1
##        self.max_prob = 0
##        self.current_prob_name = ""
        self.current_prob= -1
        self.sol_current_prob=-1
        self.current_rep=-1
##        self.max_prob_term = []
        self.unique=0
        self.sol_unique=0
##        self.current_prob_term=-1
        self.current_term =-1
        self.sol_current_term=-1
        self.func ="y**2*sin(pi*x)"
        self.DE = "F[0][0]"
        self.boundType = 0
        self.num_points = 9
        self.point1 = [0,0]
        self.point2 = [1,1]
        self.x_order = 0
        self.y_order = 0
        self.x_explicit_order=0
        self.y_explicit_order=0
        #set the system path
        progname = sys.argv[0]
        dirname = os.path.dirname(progname)
        if dirname and dirname != os.curdir:
            self.dir = dirname
            index = -1
            for i in range(len(sys.path)):
                p = sys.path[i]
                if p in ("", os.curdir):
                    index = i
            if index >= 0:
                sys.path[index] = dirname
            else:
                sys.path.insert(0, dirname)
        else:
            self.dir = os.getcwd()
        sys.path.insert(0, self.dir+'/samples')

        
    def MainNotebook(self):
        top = self.root
        w = Tix.NoteBook(top, ipadx=5, ipady=5, options="""
        *TixNoteBook*tagPadX 6
        *TixNoteBook*tagPadY 4
        *TixNoteBook*borderWidth 2
        """)
        # This may be required if there is no *Background option
        top['bg'] = w['bg']

        w.add('prob', label='Problem Parameters', underline=0,
              createcmd=lambda w=w, name='prob': self.ProblemParameters(w, name))
        w.add('sol', label='Solution Representation', underline=0,
              createcmd=lambda w=w, name='sol': self.SolutionRepresentation(w, name))
##        w.add('opt', label='Optimisation', underline=0,
##              createcmd=lambda w=w, name='opt': self.ProblemParameters(w, name))
##        w.add('exp', label='Experiment', underline=0,
##              createcmd=lambda w=w, name='exp': self.ProblemParameters(w, name))
        return w

    def MkMainMenu(self):
        top = self.root
        w = Tix.Frame(top, bd=2, relief=RAISED)
        file = Tix.Menubutton(w, text='File', underline=0, takefocus=0)
        help = Tix.Menubutton(w, text='Help', underline=0, takefocus=0)
        file.pack(side=LEFT)
        help.pack(side=RIGHT)
        fm = Tix.Menu(file, tearoff=0)
        file['menu'] = fm
        hm = Tix.Menu(help, tearoff=0)
        help['menu'] = hm

       
        if w.tk.eval ('info commands console') == "console":
            fm.add_command(label='Console', underline=1,
                           command=lambda w=w: w.tk.eval('console show'))

        fm.add_command(label='Exit', underline=1,
                     command = lambda self=self: self.quitcmd () )
        hm.add_checkbutton(label='BalloonHelp', underline=0, command=self.ToggleHelp,
                           variable=self.useBalloons)
        # The trace variable option doesn't seem to work, instead I use 'command'
        #apply(w.tk.call, ('trace', 'variable', self.useBalloons, 'w',
        #             ToggleHelp))
        return w
    def MkMainStatus(self):
        global test
        top = self.root

        w = Tix.Frame(top, relief=Tix.RAISED, bd=1)
        test.statusbar = Tix.Label(w, relief=Tix.SUNKEN, bd=1)
        test.statusbar.form(padx=3, pady=3, left=0, right='%70')
        return w
    
    def build(self):
        root = self.root
        z = root.winfo_toplevel()
        z.wm_title('Neural Network based Differential Equation experimentation engine')
        z.geometry('790x590+10+10')
        self.balloon = Tix.Balloon(root)
        frame1 = self.MkMainMenu()
        frame2 = self.MainNotebook()
        frame3 = self.MkMainStatus()
        frame1.pack(side=TOP, fill=X)
        #frame3.pack(side=BOTTOM, fill=X)
        frame2.pack(side=TOP, expand=1, fill=BOTH, padx=4, pady=4)
        self.balloon['statusbar'] = self.statusbar
        z.wm_protocol("WM_DELETE_WINDOW", lambda self=self: self.quitcmd())

    def quitcmd (self):
        """Quit our mainloop. It is up to you to call root.destroy() after."""
       # self.exit = 0
        self.root.destroy()



    def destroy (self):
        self.root.destroy()

####################Page 1: Problem Parameters##########################
##Widget hierarchy
    def ProblemParameters(self, nb, name):
        w = nb.page(name)
        prefix = Tix.OptionName(w)
        if not prefix:
            prefix = ''
        w.option_add('*' + prefix + '*TixLabelFrame*label.padX', 4)

        func = Tix.LabelFrame(w, label='functions')
        bound = Tix.LabelFrame(w, label='rectangular boundary')
        de = Tix.LabelFrame(w, label='differential operations')
        prob_type = Tix.LabelFrame(w, label='problem type')
        bc = Tix.LabelFrame(w, label='boundary conditions')
        monitor = Tix.LabelFrame(w, label='Problem set')

        self.ProbMonitor(monitor.frame)    
        self.ProbFunc(func.frame)
        self.ProbBound(bound.frame)
        self.ProbDiffEqu(de.frame)
        self.ProbType(prob_type.frame)
        self.ProbBoundCond(bc.frame)

        monitor.form(left=0, right=-1, top=0)
        func.form(left=0, right='%50',top=monitor)
        de.form(left=func, right=-1, top=monitor)
        bound.form(left=0, right='&'+str(func), top=func)
       # prob_type.form(left=func, right=-1, top=de)
        bc.form(left=func, right=-1, top=de)

        print "Welcome to the Neural Network based Differential Equation Experimentation Engine."
        print "Apply 'new' to create a new problem, then 'insert' to add consistent differential equation terms."
        print "Each term can only differ by the differential operation and the source function which is derived anyway."
        print "Once you have created your problem, proceed to the 'Solution Representation' page to bind the problem to a representation and solve via optimisation or interpolation."
    def ProbMonitor(self, w):
        de = Tix.LabelFrame(w, label='Current Differential Equations')
        bound = Tix.LabelFrame(w, label='Static Boundary Information')
        control = Tix.LabelFrame(w, label='Problem Control')

        self.ProbMonitorDiffEqu(de.frame)
        self.ProbMonitorBoundary(bound.frame)    
        self.ProbMonitorControl(control.frame)
        
        de.form(left=0, right=-1, top=0)
        bound.form(left=0, right='%60', top=de)
        control.form(left=bound, right=-1, top=de)

    def ProbType(self, w):
        self.prob_assign_option = Tix.Select(w, label='Set Problem Type:', allowzero=1, radio=1)
        self.prob_assign_option.add('de', text='Differential Equation') #command=self.setSolAssign())
        self.prob_assign_option.add('target_mapping', text='Target Mapping') 
        self.prob_assign_option.subwidget_list['de'].invoke()
        self.prob_assign_opt= self.func_assign_option["value"]
        self.prob_assign_option.pack(side=Tix.TOP,padx=5, pady=3, fill=Tix.X)
    def ProbFunc(self, w):
        self.f_list = Tix.ComboBox(w, label="Function List: ", dropdown=0,
            command=lambda w=w: self.selectFunc(w), editable=1, variable=self.func,
            options='listbox.height 3 label.padY 5 label.width 10 label.anchor ne')
        self.f_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.f_list.insert(Tix.END, 'y**2*sin(pi*x)')
        self.f_list.insert(Tix.END, '((x-0.5)**2 + (y-0.5)**2 +1)**(1/2.0)')
        self.f_list.insert(Tix.END, 'exp(-x)*(x+y**3.0)')
        self.f_list.insert(Tix.END, '16*(x-0.5)**2*(y-0.5)**2')
        self.f_list.insert(Tix.END, 'cos(x*2*pi)*sin(y*2*pi)')
        self.f_list.insert(Tix.END, 'sin(x*2*pi + y*2*pi)')
        self.f_list.insert(Tix.END, 'ln(1+(x-0.5)**2 +(y-0.5)**2)')
        self.f_list.set_silent('y**2*sin(pi*x)')
        self.f_list.pack(fill=Tix.X, padx=5, pady=3)
        x_order = Tix.IntVar()
        y_order = Tix.IntVar()
        #radio buttons decide whether the function is assigned to solution or source
        self.func_assign_option = Tix.Select(w, label='Assign function to:', allowzero=1, radio=1)
        self.func_assign_option.add('solution', text='Solution') #command=self.setSolAssign())
        self.func_assign_option.add('source', text='Source Function') 
        self.func_assign_option.subwidget_list['solution'].invoke()
        self.func_assign_opt= self.func_assign_option["value"]
        
        xn = Tix.Control(w, label='x order: ', integer=1,
                        variable=x_order, min=0, max=3,command=lambda w=w: self.select_X_order(w),
                        options='entry.width 5 label.width 6 label.anchor e')
        yn = Tix.Control(w, label='y order: ', integer=1,
                        variable=y_order, min=0, max=3,command=lambda w=w: self.select_Y_order(w),
                        options='entry.width 5 label.width 6 label.anchor e')

        self.func_assign_option.pack(side=Tix.BOTTOM,padx=5, pady=3, fill=Tix.X)

    def ProbBound(self, w):
        self.point1_lab = Tix.LabelEntry(w, label='Point 1:', options='entry.width 5')
        self.point1_lab.entry.insert(0,'[0,0]') 
        #print self.point1_lab.entry.get()
        self.point2_lab= Tix.LabelEntry(w, label='Point 2:', options='entry.width 5')
        self.point2_lab.entry.insert(0,'[1,1]')
        self.num_points_lab= Tix.LabelEntry(w, label='Number of points:', options='entry.width 5')
        self.num_points_lab.entry.insert(0,'9')
        self.point1_lab.pack(side=Tix.TOP, padx=3, pady=3, fill=Tix.BOTH)
        self.point2_lab.pack(side=Tix.TOP, padx=3, pady=3, fill=Tix.BOTH)
        self.num_points_lab.pack(side=Tix.TOP, padx=3, pady=3, fill=Tix.BOTH)
        
    def ProbDiffEqu(self, w):
        #user_fn = Tix.StringVar()
        self.diff_e = Tix.ComboBox(w, label="Differential Operation List (DOs): ", dropdown=0,
            command=lambda w=w: self.select_DE(w), editable=1, variable=self.DE,
            options='listbox.height 3 label.padY 5 label.width 25 label.anchor ne')
        self.diff_e.pack(side=Tix.TOP, anchor=Tix.W)
        self.diff_e.insert(Tix.END, 'F[0][0]')
        self.diff_e.insert(Tix.END, 'F[1][0]')
        self.diff_e.insert(Tix.END, 'F[0][1]')
        self.diff_e.insert(Tix.END, 'F[2][0]')
        self.diff_e.insert(Tix.END, 'F[0][2]')
        self.diff_e.insert(Tix.END, 'F[1][1]')
        self.diff_e.insert(Tix.END, 'F[2][1]')
        self.diff_e.insert(Tix.END, 'F[1][2]')
        self.diff_e.insert(Tix.END, 'F[3][0]')
        self.diff_e.insert(Tix.END, 'F[0][3]')
        self.diff_e.insert(Tix.END, 'F[2][0]+F[0][2]')
        self.diff_e.set_silent('F[0][0]')
        self.diff_e.pack(fill=Tix.X, padx=5, pady=3)
    def ProbBoundCond(self, w):
        #user_fn = Tix.StringVar()
        self.bc = Tix.ComboBox(w, label=" ", dropdown=0,
            command=lambda w=w: self.select_bound(w), editable=0, variable=self.boundType,
            options='listbox.height 3 label.padY 5 label.width 10 label.anchor ne')
        self.bc.pack(side=Tix.TOP, anchor=Tix.W)
        self.bc.insert(Tix.END, 'Dirichlet only')
        self.bc.insert(Tix.END, 'Arbitrary Mixed Dirichlet/Neumann')
        self.bc.insert(Tix.END, 'Random Mixed Dirichlet/Neumann')
        self.bc.insert(Tix.END, 'Neumann only')
        #implement a view function
        
        self.bc.set_silent('Dirichlet only')
        self.bc.pack(fill=Tix.X, padx=5, pady=3)
    def ProbMonitorDiffEqu(self, w):
        user_prob = Tix.StringVar()
        self.prob_list = Tix.ComboBox(w, label="Problem: ", dropdown=0,
            command=self.select_problem, editable=0, variable=user_prob,
            options='listbox.height 3 label.padY 2 label.width 8 label.anchor ne')
        self.prob_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.diff_list = Tix.ComboBox(w, label="DOs: ", dropdown=0,
            command=self.select_diff, editable=0,
            options='listbox.height 3 label.padY 2 label.width 4 label.anchor ne')
        self.diff_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.source_list = Tix.ComboBox(w, label="Source Functions: ", dropdown=0,
            command=self.select_source, editable=0,
            options='listbox.height 3 label.padY 2 label.width 13 label.anchor ne')
        self.solution_list = Tix.ComboBox(w, label="Solutions: ", dropdown=0,
            command=self.select_solution, editable=0,
            options='listbox.height 3 label.padY 2 label.width 10 label.anchor ne')

        self.view_mode_option = Tix.Select(w, label='View mode:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.view_mode_option.add('boundary', text='Boundary') #command=self.setSolAssign())
        self.view_mode_option.add('source', text='Source Function')
        self.view_mode_option.add('solution', text='Solution')
        self.view_mode_option.subwidget_list['source'].invoke()
        self.view_mode_opt= self.view_mode_option["value"]
        
        self.source_list.pack(side=Tix.LEFT, anchor=Tix.W) 
        self.prob_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.diff_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.solution_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.view_mode_option.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        
    def ProbMonitorBoundary(self, w):
        self.point1_list = Tix.ComboBox(w, label="Point 1: ", dropdown=0,
            command=self.select_point1, editable=0,
            options='listbox.height 1 label.padY 2 label.width 10 label.anchor ne')
        self.point1_list.pack(side=Tix.LEFT, anchor=Tix.W)
        self.point2_list = Tix.ComboBox(w, label="Point 2 : ", dropdown=0,
            command=self.select_point2, editable=0,
            options='listbox.height 1 label.padY 2 label.width 10 label.anchor ne')
        self.point2_list.pack(side=Tix.LEFT, anchor=Tix.W)
        self.bound_type_list = Tix.ComboBox(w, label="Boundary Type : ", dropdown=0,
            command=self.select_bound_type, editable=0,
            options='listbox.height 1 label.padY 2 label.width 15 label.anchor ne')
        self.bound_type_list.pack(side=Tix.LEFT, anchor=Tix.W)

        
    def ProbMonitorControl(self, w):
        # 4 buttons
        #new
        #insert
        #delete
        #view
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('new', text='New', underline=0, width=5,
                command=lambda w=w: self.newProblem())
        box.add('insert', text='Insert', underline=0, width=5,
                command=lambda w=w: self.insertDE())
        box.add('view', text='View', underline=0, width=5,
                command=lambda w=w: self.viewFunction())
        box.add('close', text='Delete', underline=0, width=5,
                command=lambda w=w: self.deleteDE())
##        box.add('boundary', text='Bound', underline=0, width=5,
##                command=lambda w=w: self.viewBoundary())
        box.pack(side=Tix.BOTTOM, fill=Tix.X)
        box.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)
####################Page 1: Problem Parameters##########################
##Callback functions
    def select_DE(self,w):
        self.DE = w
##        print self.DE

    def select_fn(self, w):
        pass 

    def select_diff(self,w):
        select = int(self.diff_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_source(self,w):
        select = int(self.source_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_solution(self,w):
        select = int(self.solution_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_point1(self,w):
        select = int(self.point1_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_point2(self,w):
        select = int(self.point2_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_bound_type(self,w):
        select = int(self.bound_type_list.slistbox.listbox.curselection()[0])
        self.current_term=select
        self.select_all()
    def select_all(self):
##        print "selecting term ", self.current_term
        self.diff_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.source_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.point1_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.point2_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.bound_type_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.solution_list.slistbox.listbox.selection_clear(0,last=Tix.END)
##        print "select current term ", self.current_term
        if self.current_term>-1:
            self.diff_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            self.source_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            self.point1_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            self.point2_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            self.bound_type_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            self.solution_list.slistbox.listbox.selection_set(self.current_term,last=self.current_term)
            #set silent
            Enum = ["Dirichlet only","Neumann only,",
                          "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""] 
            prob_list =  self.problems[self.current_prob].prob_list
            self.diff_list.set_silent(prob_list[self.current_term].DE)
            self.source_list.set_silent(prob_list[self.current_term].sourceFunction)
            self.point1_list.set_silent(prob_list[self.current_term].point1)
            self.point2_list.set_silent(prob_list[self.current_term].point2)
            self.bound_type_list.set_silent(Enum[prob_list[self.current_term].boundaryTypeId])
            self.solution_list.set_silent(prob_list[self.current_term].sln)
        else:
            self.diff_list.set_silent("")
            self.source_list.set_silent("")
            self.point1_list.set_silent("")
            self.point2_list.set_silent("")
            self.bound_type_list.set_silent("")   
            self.solution_list.set_silent("") 

    def select_problem(self,w):
        #load self.diff_list, self.source_list, self.point1_list, self.point2_list, self.bound_type_list           
        #find index of problem
        if len(self.problems) >0:
            self.source_list.slistbox.listbox.delete(0, END)
            self.diff_list.slistbox.listbox.delete(0, END)
            self.point1_list.slistbox.listbox.delete(0, END)
            self.point2_list.slistbox.listbox.delete(0, END)
            self.bound_type_list.slistbox.listbox.delete(0, END)
            self.solution_list.slistbox.listbox.delete(0, END)
            self.current_prob_name = w
            #find index of self.current_prob_name in self.problem_list

            self.current_prob = map(lambda x: x.name, self.problems).index(self.current_prob_name)
##            print "select problem ", w
            prob_list =  self.problems[self.current_prob].prob_list
            self.prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
            self.prob_list.slistbox.listbox.selection_set(self.current_prob,last=self.current_prob)
            #print self.current_prob, self.current_prob_name 
            #print  prob_list
            if len(prob_list)>0:
                Enum = ["Dirichlet only","Neumann only","Arbitrary Mixed Dirichlet/Neumann",
                        "Random Mixed Dirichlet/Neumann",""]
                for i,p in enumerate(prob_list):
                    #print type(p)
                    self.diff_list.insert(Tix.END, p.DE)
                    self.source_list.insert(Tix.END, p.sourceFunction)
                    self.point1_list.insert(Tix.END, p.point1)
                    self.point2_list.insert(Tix.END, p.point2)
                    self.bound_type_list.insert(Tix.END, Enum[p.boundaryTypeId])
                    self.solution_list.insert(Tix.END, p.sln)          
                self.diff_list.set_silent(prob_list[0].DE)
                self.source_list.set_silent(prob_list[0].sourceFunction)
                self.point1_list.set_silent(prob_list[0].point1)
                self.point2_list.set_silent(prob_list[0].point2)
                self.bound_type_list.set_silent(Enum[prob_list[0].boundaryTypeId])
                self.solution_list.set_silent(p.sln)
                self.current_term=0

    def newProblem(self):
        #create new instance if none exists or the current instance has more than one term
        
        if len(self.problems) == 0 or len(self.problems[self.current_prob].prob_list) > 0:
            self.problems += [ProblemInformationList(name="Problem "+str(self.unique))]
            #self.problem_list += ["Problem "+str(self.unique)]
            #print self.problems, self.problem_list, self.current_prob
            
            #self.max_prob+=1
            # the index of the current problem
            self.current_prob=len(self.problems)-1
            self.current_term=-1
            #self.max_prob_term +=[0]
            self.unique +=1
            #print self.current_prob
            self.prob_list.insert(Tix.END, self.problems[self.current_prob].name)
            self.prob_list.set_silent(self.problems[self.current_prob].name)
            #clear other combo boxes
           # print self.source_list.subwidget_list
            #select the new problem
            self.prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
            self.prob_list.slistbox.listbox.selection_set(self.current_prob,last=self.current_prob)
           
            self.source_list.slistbox.listbox.delete(0, END)
            self.diff_list.slistbox.listbox.delete(0, END)
            self.point1_list.slistbox.listbox.delete(0, END)
            self.point2_list.slistbox.listbox.delete(0, END)
            self.bound_type_list.slistbox.listbox.delete(0, END)
            self.solution_list.slistbox.listbox.delete(0, END)
            #SelCol.ShowItem(2, False)
    def insertDE(self):
        if  len(self.problems) >0:
    ##        print "insert"
            Enum = ["Dirichlet only","Neumann only",
                      "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""] 
            prob =self.problems[self.current_prob].insert()
            
            #load problem parameters
            #prob =  self.problems[self.current_prob].prob_list[self.current_term]
            self.point1 = eval(self.point1_lab.entry.get())
            self.point2 = eval(self.point2_lab.entry.get())
            self.num_points = eval(self.num_points_lab.entry.get())
            self.DE = self.diff_e.entry.get()
            self.func = self.f_list.entry.get()
            self.boundType = Enum.index(self.bc.entry.get())

  
            prob.setDomain(point1 = self.point1,point2=self.point2,numpoints=self.num_points)
            prob.setBoundaryPoints(0)
            prob.setDifferentialEquation(self.DE)
            prob.setDirichletNeumannAngles(self.boundType,numAngles=4,nstate=1)
            prob.setBoundarySegments()
            prob.setBoundaryValues(self.func)

            prob.deriveSourceFunction(self.func)
            
                
            self.diff_list.insert(Tix.END, prob.DE)
            self.source_list.insert(Tix.END, prob.sourceFunction)
            self.point1_list.insert(Tix.END, prob.point1)
            self.point2_list.insert(Tix.END, prob.point2)
            self.bound_type_list.insert(Tix.END, Enum[prob.boundaryTypeId])
            self.solution_list.insert(Tix.END, prob.sln)
            
            self.diff_list.set_silent(prob.DE)
            self.source_list.set_silent(prob.sourceFunction)
            self.point1_list.set_silent(prob.point1)
            self.point2_list.set_silent(prob.point2)
            self.bound_type_list.set_silent(Enum[prob.boundaryTypeId])
            self.solution_list.set_silent(prob.sln)

                #insert all parameters
            self.current_term=len(self.problems[self.current_prob].prob_list)-1
##            print "current term, ",  self.current_term
            self.select_all()                

    def viewFunction(self):
        self.view_mode_opt= self.view_mode_option["value"]
##        print self.view_mode_opt
        if self.view_mode_opt == 'source':
            if len(self.problems)>0 and len(self.problems[self.current_prob].prob_list)>0:
                term =  self.problems[self.current_prob].prob_list[self.current_term]
                term.viewSourceFunction()
        elif self.view_mode_opt == 'boundary':
            self.viewBoundary()
        elif self.view_mode_opt == 'solution':
            self.viewSolution()

    def viewSolution(self):
        if len(self.problems)>0 and len(self.problems[self.current_prob].prob_list)>0:
            term =  self.problems[self.current_prob].prob_list[self.current_term]
            term.viewSolution()
    
    def viewBoundary(self):
        if len(self.problems)>0 and len(self.problems[self.current_prob].prob_list)>0:
            term =  self.problems[self.current_prob].prob_list[self.current_term]
            term.viewBoundary()
        #view selected part of instance
    
    def deleteDE(self):
##        print "delete"
        #delete selected part of instance
        
        if  len(self.problems) >0:
            prob =self.problems[self.current_prob]
            if len(prob.prob_list)>0:#self.current_term > -1:
                print "delete", self.current_term, len(prob.prob_list) 
                prob.delete(self.current_term)
                self.source_list.slistbox.listbox.delete(self.current_term, self.current_term)
                self.diff_list.slistbox.listbox.delete(self.current_term, self.current_term)
                self.point1_list.slistbox.listbox.delete(self.current_term, self.current_term)
                self.point2_list.slistbox.listbox.delete(self.current_term, self.current_term)
                self.bound_type_list.slistbox.listbox.delete(self.current_term, self.current_term)
                self.solution_list.slistbox.listbox.delete(self.current_term, self.current_term)
                #if self.current_term>-1:
                self.current_term =len(prob.prob_list)-1
                self.select_all()
            else:
##                print "problem, no terms"
##                print "num probs ", len(self.problems)
                self.problems = self.problems[0:self.current_prob]+self.problems[self.current_prob+1:]
##                print "num probs ", len(self.problems)
                self.prob_list.slistbox.listbox.delete(self.current_prob, self.current_prob)
                self.prob_list.set_silent("")
                self.current_prob=len(self.problems)-1
                if self.current_prob>-1:
##                    print "name ", self.problems[self.current_prob].name
                    self.select_problem(self.problems[self.current_prob].name)
                    self.current_term = len(self.problems[self.current_prob].prob_list)-1
                    self.select_all()
                self.prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
                self.prob_list.slistbox.listbox.selection_set(self.current_prob,last=self.current_prob)
    

    def setSolAssign(self):
        self.func_assign_opt = "solution"
    def setSourceAssign(self):
        self.func_assign_opt = "source"
    def toggleFuncAssignment(self,w):
        self.func_assign_opt = w
        
    def select_X_order(self,w):
        self.x_order = w
##        print self.x_order
        
    def select_Y_order(self,w):
        self.y_order = w
##        print self.y_order
        #w.pack(side=Tix.TOP, fill=Tix.BOTH, expand=0)
        
    def selectFunc(self, w):
     #   print "here"
        #print self.func
        self.func = w
##        print self.func

    def differentiate_fn(self, event=None):
        # tixDemo:Status "Year = %s" % demo_year.get()
        x=Symbol('x')
        y=Symbol('y')
        func=self.func
        fn=diff(eval(func),x,int(self.x_order))
        fn=diff(fn,y,int(self.y_order))
        self.func = str(fn)
        self.func=self.func.replace("1/2", "1.0/2")
        self.f_list.set_silent(self.func)
##        print self.func
    def select_bound(self, w, event=None):
        # tixDemo:Status "Year = %s" % demo_year.get()
        Enum = ["Dirichlet only","Neumann only",
                  "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""]
        self.boundType = Enum.index(w)
##        print self.boundType
        pass
    def MkDirListWidget(self, w):
        msg = Tix.Message(w, 
                  relief=Tix.FLAT, width=240, anchor=Tix.N,
                  text='The TixDirList widget gives a graphical representation of the file system directory and makes it easy for the user to choose and access directories.')
        dirlist = Tix.DirList(w, options='hlist.padY 1 hlist.width 25 hlist.height 16')
        msg.pack(side=Tix.TOP, expand=1, fill=Tix.BOTH, padx=3, pady=3)
        dirlist.pack(side=Tix.TOP, padx=3, pady=3)

    def MkExFileWidget(self, w):
        msg = Tix.Message(w, 
                  relief=Tix.FLAT, width=240, anchor=Tix.N,
                  text='The TixExFileSelectBox widget is more user friendly than the Motif style FileSelectBox.')
        # There's a bug in the ComboBoxes - the scrolledlistbox is destroyed
        box = Tix.ExFileSelectBox(w, bd=2, relief=Tix.RAISED)
        msg.pack(side=Tix.TOP, expand=1, fill=Tix.BOTH, padx=3, pady=3)
        box.pack(side=Tix.TOP, padx=3, pady=3)
    def ToggleHelp(self):
        if demo.useBalloons.get() == '1':
            demo.balloon['state'] = 'both'
        else:
            demo.balloon['state'] = 'none'

####################Page 2: Representation##########################
## Widget Hierarhcy 
    def SolutionRepresentation(self, nb, name):
##        print "solution"
        w = nb.page(name)
        prefix = Tix.OptionName(w)
        if not prefix:
            prefix = ''
        w.option_add('*' + prefix + '*TixLabelFrame*label.padX', 4)

        rep_monitor = Tix.LabelFrame(w, label='Representation Workbench')
##        de = Tix.LabelFrame(w, label='Current Differential Equations')
##        bound = Tix.LabelFrame(w, label='Static Boundary Information')
        rep_prob = Tix.LabelFrame(w, label='Problem options')
        rep_choices = Tix.LabelFrame(w, label='Representation Options')
        rep_explicit_view_opts = Tix.LabelFrame(w, label='Viewing Options')  
        rep_control = Tix.LabelFrame(w, label='Representation Control')
        rep_prob_view = Tix.LabelFrame(w, label='Problem View')
        rep_network_options = Tix.LabelFrame(w, label='Network Options')
        rep_misc= Tix.LabelFrame(w, label='Misc. Options')
        opt_param = Tix.LabelFrame(w, label='Method of Solution Parameters')
        opt_control =  Tix.LabelFrame(w, label='Method of Solution Control')
        opt_results = Tix.LabelFrame(w, label='Method of Solution Results')

        self.SolutionRepresentationMonitor(rep_monitor.frame)
        self.SolutionProb(rep_prob.frame)
        self.SolutionChoicesSelection(rep_choices.frame)    # a
        self.SolutionControl(rep_control.frame)
        self.SolutionExplicitViewOptions(rep_explicit_view_opts.frame)
        self.SolutionProblemView(rep_prob_view.frame)
        self.SolutionNetworkOptions(rep_network_options.frame)
        self.SolutionMiscOptions(rep_misc.frame)
        self.SolutionOptimisationParam(opt_param.frame)
        self.SolutionOptimisationControl(opt_control.frame)
        self.SolutionOptimisationResults(opt_results.frame)   
        
        
        rep_monitor.form(left=0, right=-1, top=0)
        rep_prob.form(left=0, right=-1, top=rep_monitor)
        rep_choices.form(left=0, right="%33", top=rep_prob)
        rep_explicit_view_opts.form(left="%33", right="%66", top=rep_prob)
        rep_control.form(left="%66", right=-1, top=rep_prob)
        rep_network_options.form(left=0, right="%33", top=rep_choices)
        rep_misc.form(left=0, right="%33", top=rep_network_options)
        rep_prob_view.form(left=rep_explicit_view_opts, right=-1, top=rep_control)
        #rep_optim.form(left =rep_misc , right=-1, top=rep_explicit_view_opts)
        opt_param.form(left=rep_misc, right="%55", top=rep_explicit_view_opts)
        opt_control.form(left=opt_param, right="%77", top=rep_explicit_view_opts)
        opt_results.form(left=opt_control, right=-1, top=rep_explicit_view_opts)
        self.sol_defined=1

    def SolutionOptimisationControl(self, w):
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('numerical', text='Optimisation', underline=0, width=8,
                command=lambda w=w: self.performOptimisation())
        box.add('analytical', text='Interpolation', underline=0, width=8,
                command=lambda w=w: self.analytical())
##        box.add('view_error', text='Error(t)', underline=0, width=6,
##                command=lambda w=w: self.viewErrorPlot())
##        box.add('boundary', text='Bound', underline=0, width=5,
##                command=lambda w=w: self.viewBoundary())
        box.pack(side=Tix.BOTTOM, fill=Tix.X)
        box.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)

    def SolutionOptimisationResults(self,w):
        self.results_list = Tix.ScrolledListBox(w)
        self.results_list.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)
    
    def SolutionOptimisationParam(self,w):
        #self.num_samples=1
        sample_no = Tix.IntVar()
        run_no = Tix.IntVar()
        self.select_samples = Tix.Control(w, label='Samples: ', integer=1,
                 min=1, max=2000,step=10,variable=sample_no,command=lambda w=w: self.select_number_samples(w),
                options='entry.width 5 label.width 20 label.anchor e')
        self.select_samples['value'] = self.num_samples
       # self.num_runs=1
        self.select_runs = Tix.Control(w, label='Alg. runs: ', integer=1,
                 min=1, max=50,step=5,variable=run_no,command=lambda w=w: self.select_number_runs(w),
                options='entry.width 5 label.width 20 label.anchor e')    
        self.select_runs['value'] = self.num_runs
 
##        self.wtf = Tix.Control(w, label='Alg. runs: ', integer=1,
##                 min=1, max=50,step=5,variable=self.num_runs,command=lambda w=w: self.select_number_runs(w),
##                options='entry.width 5 label.width 20 label.anchor e')    
##        self.wtf['value'] = self.num_runs
        
        self.independent_samples_opt = Tix.Checkbutton(w,  text="Independent samples",
                command=lambda w=w: self.SolutionToggleSampleIndependence())
        self.independent_samples_opt.invoke() 
        self.global_search_option = Tix.Checkbutton(w,  text="Global search",
                command=lambda w=w: self.SolutionToggleGlobalSearch())
        self.global_search_option.invoke()

        self.select_samples.pack(side=Tix.TOP, anchor=Tix.W)
        self.independent_samples_opt.pack(side=Tix.TOP, anchor=Tix.W)
        self.select_runs.pack(side=Tix.TOP, anchor=Tix.W)
##        self.wtf.pack(side=Tix.TOP, anchor=Tix.W)
        self.global_search_option.pack(side=Tix.TOP, anchor=Tix.W)
    def viewErrorPlot(self):
        pass
        
    def SolutionNetworkOptions(self, w):
        #Number of nodes
        self.select_num_nodes = Tix.Control(w, label='Number of nodes: ', integer=1,
                 min=1, max=50,step=5,variable=self.num_nodes,command=lambda w=w: self.select_number_nodes(w),
                options='entry.width 5 label.width 20 label.anchor e')
        self.select_num_nodes['value'] = self.num_nodes
        #button box
        self.network_param_option = Tix.Select(w, label='Assign Parameters:',\
                                                allowzero=1, radio=1,orientation=Tix.HORIZONTAL)
        self.network_param_option.add('zero_network', text='Zero Network') #command=self.setSolAssign())
        self.network_param_option.add('random_network', text='Random Network')
        self.network_param_option.subwidget_list['zero_network'].invoke()
        self.select_num_nodes.pack(side=Tix.TOP, anchor=Tix.W) 
        self.network_param_option.pack(side=Tix.TOP, padx=5, pady=3, fill=Tix.X)
        
            
    def SolutionRepresentationMonitor(self, w):

        self.sol_rep_list = Tix.ComboBox(w, label="Rep:", dropdown=0,
            command=self.select_sol_representation, editable=0, 
            options='listbox.height 2 label.padY 2 label.width 5 label.anchor ne')
        
        self.sol_rep_info_list = Tix.ComboBox(w, label="Rep info:", dropdown=0,
            command=self.select_sol_rep_info_list, editable=0,
            options='listbox.height 2 label.padY 2 label.width 7 label.anchor ne')
        self.sol_mon_prob_list = Tix.ComboBox(w, label="Prob:", dropdown=0,
            command=self.select_sol_rep_prob_list, editable=0,
            options='listbox.height 2 label.padY 2 label.width 5 label.anchor ne')
        self.sol_mon_prob_info_list = Tix.ComboBox(w, label="Prob info:", dropdown=0,
            command=self.select_sol_rep_problem_info_list, editable=0,
            options='listbox.height 2 label.padY 2 label.width 7 label.anchor ne')
        self.sol_prob_mon_DE_list = Tix.ComboBox(w, label="DOs:", dropdown=0,
            command=self.select_sol_rep_problem_info_list, editable=0,
            options='listbox.height 2 label.padY 2 label.width 3 label.anchor ne')
        self.sol_prob_mon_source_list = Tix.ComboBox(w, label="Source:", dropdown=0,
            command=self.select_sol_rep_problem_info_list, editable=0,
            options='listbox.height 2 label.padY 2 label.width 5 label.anchor ne')
        self.sol_rep_list.pack(side=Tix.LEFT, anchor=Tix.W)
        self.sol_rep_info_list.pack(side=Tix.LEFT, anchor=Tix.W) 
        self.sol_mon_prob_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_mon_prob_info_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_prob_mon_DE_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_prob_mon_source_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        
        
    def SolutionProb(self, w):
        de = Tix.LabelFrame(w, label='Current Problems')
        #bound = Tix.LabelFrame(w, label='Boundary Information')

        self.SolutionDiffEqu(de.frame)
        #self.SolutionBoundary(bound.frame)    
        
        de.pack(side=Tix.TOP, anchor=Tix.W)
       # bound.pack(side=Tix.TOP, anchor=Tix.W)

    def SolutionDiffEqu(self, w):
        self.sol_prob_list = Tix.ComboBox(w, label="Problem:", dropdown=0,
            command=self.select_sol_problem, editable=0,
            options='listbox.height 2 label.padY 2 label.width 8 label.anchor ne')
        self.sol_prob_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.sol_diff_list = Tix.ComboBox(w, label="DOs:", dropdown=0,
            command=self.select_sol_diff, editable=0,
            options='listbox.height 2 label.padY 2 label.width 3 label.anchor ne')
        self.sol_diff_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.sol_source_list = Tix.ComboBox(w, label="Source:", dropdown=0,
            command=self.select_sol_source, editable=0,
            options='listbox.height 2 label.padY 2 label.width 5 label.anchor ne')
        self.sol_solution_list = Tix.ComboBox(w, label="Solutions: ", dropdown=0,
            command=self.select_sol_solution, editable=0,
            options='listbox.height 2 label.padY 2 label.width 10 label.anchor ne')
        self.sol_bound_type_list = Tix.ComboBox(w, label="Boundary Type:", dropdown=0,
            command=self.select_sol_bound_type, editable=0,
            options='listbox.height 2 label.padY 2 label.width 12 label.anchor ne')
        self.sol_point_list = Tix.ComboBox(w, label="Points: ", dropdown=0,
            command=self.select_sol_point, editable=0,
            options='listbox.height 2 label.padY 2 label.width 5 label.anchor ne')
        self.sol_source_list.pack(side=Tix.LEFT, anchor=Tix.W) 
        self.sol_prob_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_diff_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_solution_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.sol_bound_type_list.pack(side=Tix.LEFT, padx=5, pady=3,fill=Tix.X)
        self.sol_point_list.pack(side=Tix.LEFT, padx=5, pady=3,fill=Tix.X)
        if len(self.problems) >0:
            #delete the problems in the problem list
            self.sol_prob_list.slistbox.listbox.delete(0, END)
            #insert
            for i,p in enumerate(self.problems):
                self.sol_prob_list.insert(Tix.END, self.problems[i].name)
                
##    def SolutionBoundary(self, w):
##        self.sol_point1_list = Tix.ComboBox(w, label="Point 1: ", dropdown=0,
##            command=self.select_sol_point1, editable=0,
##            options='listbox.height 1 label.padY 2 label.width 10 label.anchor ne')
##        self.sol_point1_list.pack(side=Tix.LEFT, anchor=Tix.W)
##        self.sol_point2_list = Tix.ComboBox(w, label="Point 2 : ", dropdown=0,
##            command=self.select_sol_point2, editable=0,
##            options='listbox.height 1 label.padY 2 label.width 10 label.anchor ne')
##        self.sol_point2_list.pack(side=Tix.LEFT, anchor=Tix.W)
##        self.sol_bound_type_list = Tix.ComboBox(w, label="Boundary Type : ", dropdown=0,
##            command=self.select_sol_bound_type, editable=0,
##            options='listbox.height 1 label.padY 2 label.width 15 label.anchor ne')
##        self.sol_bound_type_list.pack(side=Tix.LEFT, anchor=Tix.W)
                                      
    def SolutionChoicesSelection(self,w):
        self.basis_function_option = Tix.Select(w, label='Basis Function:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.basis_function_option.add('sigmoid', text='Sigmoid') #command=self.setSolAssign())
        self.basis_function_option.add('sinusoid', text='Sinusoid')
        self.basis_function_option.add('radial', text='Radial')
        
        self.bound_representation_option = Tix.Select(w, label='Boundary Representation Type:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.bound_representation_option.add('explicit', text='Explicit') #command=self.setSolAssign())
        self.bound_representation_option.add('implicit', text='Implicit')
##    def RepresentationProblemFilterSelection(self,w):

        self.basis_function_option.subwidget_list['sigmoid'].invoke()
        self.bound_representation_option.subwidget_list['explicit'].invoke()
         
        self.basis_function_option.pack(side=Tix.LEFT, anchor=Tix.W)
        self.bound_representation_option.pack(side=Tix.LEFT, anchor=Tix.W)


    def SolutionMiscOptions(self,w):
        self.interpolation_option = Tix.Select(w, label='Explicit: Boundary interpolation method:', allowzero=1, radio=1,orientation=Tix.HORIZONTAL)
        self.interpolation_option.add('tps', text='TPS') #command=self.setSolAssign())
        self.interpolation_option.add('smoothing', text='Smoothing')
        self.interpolation_option.subwidget_list['smoothing'].invoke()
        self.interpolation_option.pack(side=Tix.TOP, anchor=Tix.W)

        self.implict_BC_opt = Tix.Checkbutton(w,  text="Implicit: Use Boundary Conditions in Energy Function",
                                              command=lambda w=w: self.SolutionToggleImplict_BC_opt())

        self.useFirstTerm_opt = Tix.Checkbutton(w,  text="Optimise with first term only",
                                              command=lambda w=w: self.SolutionToggleFirstTerm_opt())
        self.implict_BC_opt.invoke()
        
        self.implict_BC_opt.pack(side=Tix.TOP, anchor=Tix.W)
        self.useFirstTerm_opt .pack(side=Tix.TOP, anchor=Tix.W)


        
        #print c["value"]

        
    def SolutionExplicitViewOptions(self, w):
        self.explicit_viewing_option = Tix.Select(w, label='Explicit Viewing Options:', allowzero=1, radio=1,orientation=Tix.HORIZONTAL)
        self.explicit_viewing_option.add('ad', text='Ad') #command=self.setSolAssign())
        self.explicit_viewing_option.add('ld', text='Ld')
        self.explicit_viewing_option.add('g', text='g')
        self.explicit_viewing_option.add('lm', text='Lm')
        self.explicit_viewing_option.add('nx', text='nx')
        self.explicit_viewing_option.add('ny', text='ny')
        self.explicit_viewing_option.add('bound', text='bound')
        self.explicit_viewing_option.subwidget_list['ad'].invoke()
        self.explicit_viewing_opt = self.explicit_viewing_option["value"]
        x_explicit_order = Tix.IntVar()
        y_explicit_order = Tix.IntVar()
        self.xn_exp = Tix.Control(w, label='x order: ', integer=1,
                        variable=x_explicit_order, min=0, max=4,command=lambda w=w: self.select_X_explicit_order(w),
                        options='entry.width 5 label.width 6 label.anchor e')
        self.yn_exp  = Tix.Control(w, label='y order: ', integer=1,
                        variable=y_explicit_order, min=0, max=4,command=lambda w=w: self.select_Y_explicit_order(w),
                        options='entry.width 5 label.width 6 label.anchor e')
        
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('error', text='residual plot', underline=0, width=10,
                command=lambda w=w: self.sol_view_initial_error())
        box.add('solution', text='Solution', underline=0, width=8,
                command=lambda w=w: self.sol_view_solution())
        box.add('dirichlet_only_network', text='Dirichlet network', underline=0, width=15,
                command=lambda w=w: self.sol_view_required_network(w))
        box.add('network', text='Network', underline=0, width=10,
                command=lambda w=w: self.sol_view_network())
        self.explicit_viewing_option.pack(side=Tix.TOP, anchor=Tix.W)
        
        self.xn_exp.pack(side=Tix.TOP, padx=5, pady=3, fill=Tix.X)
        self.yn_exp.pack(side=Tix.TOP, padx=5, pady=3, fill=Tix.X)
        box.pack(side=Tix.TOP, padx=5, pady=3, fill=Tix.X)                    
        self.init_error = Tix.LabelEntry(w, label='log_10 Error:', options='entry.width 20')
        self.init_error.entry.insert(0,'0')
        
        self.init_error.pack(side=Tix.TOP, padx=5, pady=3, fill=Tix.X)
        
    def select_fn(self):
        pass
                           
    def SolutionControl(self, w):
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('new', text='New', underline=0, width=5,
                command=lambda w=w: self.newRepresentation())
        box.add('view', text='View', underline=0, width=5,
                command=lambda w=w: self.view_Mcfall_rep())
        box.add('delete', text='Delete', underline=0, width=5,
                command=lambda w=w: self.deleteRep())
        box.add('update', text='Update', underline=0, width=5,
                command=lambda w=w: self.updateRep())
##        box.add('boundary', text='Bound', underline=0, width=5,
##                command=lambda w=w: self.viewBoundary())
        box.pack(side=Tix.BOTTOM, fill=Tix.X)
        box.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)

    def SolutionProblemView(self,w):
        self.view_solution_prob = Tix.Select(w, label='View mode:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.view_solution_prob.add('boundary', text='Boundary') #command=self.setSolAssign())
        self.view_solution_prob.add('source', text='Source Function')
        self.view_solution_prob.add('solution', text='Solution')
        self.view_solution_prob.subwidget_list['source'].invoke()
        self.view_solution_prob_val = self.view_solution_prob["value"]

        #self.view_solution_prob= self.view_solution_prob["value"] 
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('view', text='View', underline=0, width=5,
                command=lambda w=w: self.viewSolFunction())
        self.view_solution_prob.pack(side=Tix.LEFT, fill=Tix.X)
        box.pack(side=Tix.LEFT, fill=Tix.X)
##        point1_label = Tix.Label(w, text='Point 1:')
##    def select_representation(self,w):
##        print "selecting problem ", w
##        
##        #load self.diff_list, self.source_list, self.point1_list, self.point2_list, self.bound_type_list           
##        #find index of problem
##        if len(self.problems) >0:
##            self.rep_type_list.slistbox.listbox.delete(0, END)
##            self.basis_list.slistbox.listbox.delete(0, END)
##            self.rep_problem_list.slistbox.listbox.delete(0, END)
##            self.rep_solution_list.slistbox.listbox.delete(0, END)
##            self.current_prob_name = w
##            #find index of self.current_prob_name in self.problem_list
##            #print map(lambda x:x.name, self.problems[self.current_prob].prob_list)
##            self.current_prob = map(lambda x: x.name, self.problems).index(self.current_prob_name)
##            print "select problem ", w
##            prob_list =  self.problems[self.current_prob].prob_list
##            self.prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
##            self.prob_list.slistbox.listbox.selection_set(self.current_prob,last=self.current_prob)
##            #print self.current_prob, self.current_prob_name 
##            #print  prob_list
##            if len(prob_list)>0:
##                Enum = ["Dirichlet only","Neumann only","Arbitrary Mixed Dirichlet/Neumann",
##                        "Random Mixed Dirichlet/Neumann",""]
##
##                for i,p in enumerate(prob_list):
##                    #print type(p)
##                    self.diff_list.insert(Tix.END, p.DE)
##                    self.source_list.insert(Tix.END, p.sourceFunction)
##                    self.point1_list.insert(Tix.END, p.point1)
##                    self.point2_list.insert(Tix.END, p.point2)
##                    self.bound_type_list.insert(Tix.END, Enum[p.boundaryTypeId])
##                    self.solution_list.insert(Tix.END, p.sln)
##                    
##                self.diff_list.set_silent(prob_list[0].DE)
##                self.source_list.set_silent(prob_list[0].sourceFunction)
##                self.point1_list.set_silent(prob_list[0].point1)
##                self.point2_list.set_silent(prob_list[0].point2)
##                self.bound_type_list.set_silent(Enum[prob_list[0].boundaryTypeId])
##                self.solution_list.set_silent(p.sln)
##                self.current_term=0


####################Page 2: Representation##########################
##Callback functions
    def SolutionToggleFirstTerm_opt(self):
##        print self.FirstTerm_option
        self.FirstTerm_option= not self.FirstTerm_option    
    def SolutionToggleImplict_BC_opt(self):
        self.implict_BC_option= not self.implict_BC_option
    def SolutionToggleGlobalSearch(self):
        self.global_search_opt= not self.global_search_opt
        
    def SolutionToggleSampleIndependence(self):
        self.sample_independent= not self.sample_independent

    def select_number_samples(self,w):
##        print "number of samples", int(w)
        self.num_samples = int(w)

    def select_number_runs(self,w):
##        print "number of runs", int(w)
        self.num_runs = int(w)

    def select_wtf(self,w):
        pass
    def select_number_nodes(self,w):
        self.num_nodes = int(w)
    def sol_view_required_network(self,w):
##        print "view required network ", self.x_explicit_order,self.y_explicit_order
        if self.current_rep>-1 and self.representations[self.current_rep].representation_type == "explicit" \
           and self.representations[self.current_rep].mcfall.mixed == False:
            rep = self.representations[self.current_rep]
            Solution.viewRequiredDirichletNetwork(self.x_explicit_order,self.y_explicit_order,rep.prob,rep.mcfall)
        #self.representations[self.current_rep].viewRequiredDirichletNetwork(self.x_explicit_order,self.y_explicit_order)
        
    def updateRep(self):
##        print "updating" ,self.problems[0].name
        if len(self.problems)>0:
            self.select_sol_problem(self.problems[0].name)
        pass
    
    def view_Mcfall_rep(self):
        ##printing the representation component
##        print self.current_rep
        if self.current_rep>-1 and self.representations[self.current_rep].representation_type == "explicit":
            component = self.explicit_viewing_option["value"]
##            print component, self.x_explicit_order,self.y_explicit_order
            self.representations[self.current_rep].mcfall.viewComponent(component,int(self.x_explicit_order),int(self.y_explicit_order))
        
        #view selected part of ins
    def viewSolFunction(self):
##        print "viewing"
        self.view_solution_prob_val= self.view_solution_prob["value"]
        if self.view_solution_prob_val == 'source':
            if len(self.problems)>0 and len(self.problems[self.sol_current_prob].prob_list)>0:
                term =  self.problems[self.sol_current_prob].prob_list[self.sol_current_term]
                term.viewSourceFunction()
        elif self.view_solution_prob_val == 'boundary':
            if len(self.problems)>0 and len(self.problems[self.current_prob].prob_list)>0:
                term =  self.problems[self.sol_current_prob].prob_list[self.sol_current_term]
                term.viewBoundary()
        elif self.view_solution_prob_val == 'solution':
            if len(self.problems)>0 and len(self.problems[self.current_prob].prob_list)>0:
                term =  self.problems[self.sol_current_prob].prob_list[self.sol_current_term]
                term.viewSolution()

    def deleteRep(self):
        if len(self.representations) >0:
##            print "problem, no terms"
##            print "num probs ", len(self.representations)
            self.representations = self.representations[0:self.current_rep]+self.problems[self.current_rep+1:]
##            print "num probs ", len(self.representations)
            self.sol_rep_list.slistbox.listbox.delete(self.current_rep, self.current_rep)
            self.sol_rep_info_list.slistbox.listbox.delete(0, last=Tix.END)
            self.sol_mon_prob_info_list.slistbox.listbox.delete(0,last=Tix.END)
            self.sol_mon_prob_list.slistbox.listbox.delete(0,last=Tix.END)
            self.sol_prob_mon_DE_list.slistbox.listbox.delete(0,last=Tix.END)
            self.sol_prob_mon_source_list.slistbox.listbox.delete(0,last=Tix.END)

            self.sol_rep_list.set_silent("")
            self.current_rep=len(self.representations)-1
            if self.current_rep>-1:
                self.select_sol_representation(self.representations[self.current_rep].name)
            self.sol_rep_list.slistbox.listbox.selection_clear(0,last=Tix.END)
            self.sol_rep_list.slistbox.listbox.selection_set(self.current_rep,last=self.current_rep)
    

    def newRepresentation(self):
        print "*************************************************"
        print "Creating new representation"
        #ensure problem is consistent, all terms must have the same solution, bounday condition set, point1, point2, num_points
        if self.sol_current_prob>-1:
            prob = self.problems[self.sol_current_prob]
            sln = prob.prob_list[0].sln
            boundaryTypeId = prob.prob_list[0].boundaryTypeId
            point1 = prob.prob_list[0].point1
            point2 = prob.prob_list[0].point2
            numpoints = prob.prob_list[0].numpoints
            consistency_flag=True
            for t in prob.prob_list:
                if t.sln == sln and t.boundaryTypeId==boundaryTypeId and t.point1==point1 and t.point2==point2 and t.numpoints == numpoints:
                    consistency_flag=True
                else:
                    consistency_flag=False
                    break
            if consistency_flag:
                self.basis_function_opt = self.basis_function_option['value']
                self.bound_representation_opt =self.bound_representation_option['value']
##                print "implicit BC option",self.implict_BC_option
                param=np.array([0,0,0,0])
                if self.network_param_option['value'] == 'zero_network':
                    param = np.array([0]*4*self.num_nodes)
                else:
                    param = self.generateRandomParameters()

##                print param
                self.representations += [RepresentationProblem(prob,name="Representation "+str(self.sol_unique),\
                                                               basis_function = self.basis_function_opt,\
                                                               representation_type = self.bound_representation_opt,\
                                                               param=param, implicit_include_BC = self.implict_BC_option)]
                #self.problem_list += ["Problem "+str(self.unique)]
                #print self.problems, self.problem_list, self.current_prob
                self.current_rep=len(self.representations)-1
                #if explicit, then create a mcfall object
                if self.bound_representation_option['value'] == 'explicit':
                    prob = self.problems[self.sol_current_prob].prob_list[0]

                    max_xorder=0
                    max_yorder=0
                    for t in self.problems[self.sol_current_prob].prob_list:
                        max_xorder = max(max_xorder,t.orderX)
                        max_yorder = max(max_yorder,t.orderY)
##                        print "max x,y ", max_xorder,max_yorder
##                    print "max x,y ", max_xorder,max_yorder
##                    print self.interpolation_option["value"]
                    
                    mcfall = McFall(prob,xn=max_xorder,yn=max_yorder,interpolation_type =self.interpolation_option["value"])
                    self.representations[self.current_rep].setMcfall(mcfall)
                    self.representations[self.current_rep].interpolation_option = self.interpolation_option["value"]
                #self.max_prob+=1
                # the index of the current problem
                
                #self.max_prob_term +=[0]
                self.sol_unique +=1

                self.sol_rep_list.slistbox.listbox.insert(Tix.END, self.representations[self.current_rep].name)            
                self.sol_rep_list.set_silent(self.representations[self.current_rep].name)
                self.select_sol_representation(self.representations[self.current_rep].name)
##            #print self.current_prob
##            self.sol_rep_list.set_silent(self.representations[self.current_rep].name)
##            #clear other combo boxes
##           # print self.source_list.subwidget_list
##            #select the new problem
##            self.prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
##            self.prob_list.slistbox.listbox.selection_set(self.current_prob,last=self.current_prob)
##           
##            self.source_list.slistbox.listbox.delete(0, END)
##            self.diff_list.slistbox.listbox.delete(0, END)
##            self.point1_list.slistbox.listbox.delete(0, END)
##            self.point2_list.slistbox.listbox.delete(0, END)
##            self.bound_type_list.slistbox.listbox.delete(0, END)
##            self.solution_list.slistbox.listbox.delete(0, END)
##            #SelCol.ShowItem(2, False)
##            print "new problem ",self.current_prob
    def generateRandomParameters(self):
        ##use a random distribution
        std = [20,10,5,5]
        param = []
        for i in range(self.num_nodes):
            for j in range(4):
                param += [(1-2*random.random())*std[j]]
        param = np.array(param)
        return param
        
        
        
    def select_sol_representation(self,w):
##        print "selecting representation", w
        if len(self.representations) >0:
            #delete the problems in the problem list      
            self.sol_rep_info_list.slistbox.listbox.delete(0, END)
            self.sol_mon_prob_list.slistbox.listbox.delete(0, END)
            self.sol_mon_prob_info_list.slistbox.listbox.delete(0, END)
            self.sol_prob_mon_DE_list.slistbox.listbox.delete(0, END)
            self.sol_prob_mon_source_list.slistbox.listbox.delete(0, END)
            self.sol_current_rep_name = w
            #find index of self.current_prob_name in self.problem_list
            #print map(lambda x:x.name, self.problems[self.current_prob].prob_list)
##            print "sol_current_rep_name ", self.sol_current_rep_name, " len(self.representations) ",len(self.representations)
##            print map(lambda x: x.name, self.representations)
            self.current_rep = map(lambda x: x.name, self.representations).index(self.sol_current_rep_name)
##            print "select problem ", w
            sol_rep =  self.representations[self.current_rep]
            self.sol_rep_list.slistbox.listbox.selection_clear(0,last=Tix.END)
            self.sol_rep_list.slistbox.listbox.selection_set(self.current_rep,last=self.current_rep)
            #print self.current_prob, self.current_prob_name 
            #print  prob_list

            Enum = ["Dirichlet only","Neumann only","Arbitrary Mixed Dirichlet/Neumann",
                    "Random Mixed Dirichlet/Neumann",""]
                    #print type(p)
            self.sol_rep_info_list.insert(Tix.END, sol_rep.basis_function)
            self.sol_rep_info_list.insert(Tix.END, sol_rep.representation_type)
            self.sol_rep_info_list.insert(Tix.END, "Nodes no. " + str(len(sol_rep.param)/4))
            if sol_rep.representation_type=='explicit':
                self.sol_rep_info_list.insert(Tix.END, sol_rep.interpolation_option)

            
            self.sol_mon_prob_list.insert(Tix.END, sol_rep.probs.name)
            self.sol_mon_prob_info_list.insert(Tix.END, "solution: " +sol_rep.probs.prob_list[0].sln)
            self.sol_mon_prob_info_list.insert(Tix.END, Enum[sol_rep.probs.prob_list[0].boundaryTypeId])
            self.sol_mon_prob_info_list.insert(Tix.END, "domain: " + str([sol_rep.probs.prob_list[0].point1,sol_rep.probs.prob_list[0].point2]))
            self.sol_mon_prob_info_list.insert(Tix.END, "num of points: " + str(sol_rep.probs.prob_list[0].numpoints))

            self.init_error.entry.delete(0,'end')
            self.init_error.entry.insert(0,str(math.log(sol_rep.error)/math.log(10)))
            for t in sol_rep.probs.prob_list:
                self.sol_prob_mon_DE_list.insert(Tix.END, t.DE)
                self.sol_prob_mon_source_list.insert(Tix.END, t.sourceFunction)
##                self.sol_diff_list.set_silent(sol_prob_list[0].DE)
##                self.sol_source_list.set_silent(sol_prob_list[0].sourceFunction)
##                self.sol_point_list.set_silent([sol_prob_list[0].point1,sol_prob_list[0].point2])
##                self.sol_bound_type_list.set_silent(Enum[sol_prob_list[0].boundaryTypeId])
##                self.sol_solution_list.set_silent(sol_prob_list[0].sln)
##                self.sol_current_term=0
        pass

    def select_sol_rep_info_list(self,w):
        pass 

    def select_sol_rep_prob_list(self,w):
        pass

    def select_sol_rep_problem_info_list(self,w):
        pass

    def select_DE(self,w):
        self.DE = w
##        print self.DE

    def select_fn(self, w):
        pass 
    def select_X_explicit_order(self,w):
        self.x_explicit_order = int(w)
##        print "select_X_explicit_order"
        if self.x_explicit_order+self.y_explicit_order>4:
##            print "restriction"
            self.x_explicit_order = 4-self.y_explicit_order
            self.xn_exp['value'] = self.x_explicit_order
##        print self.x_explicit_order
        
    def select_Y_explicit_order(self,w):
        self.y_explicit_order = int(w)
        if self.y_explicit_order+self.x_explicit_order>4:
            self.y_explicit_order = 4-self.x_explicit_order
            self.yn_exp['value'] = self.y_explicit_order
##        print self.y_explicit_order
    def select_sol_diff(self,w):
        select = int(self.sol_diff_list.slistbox.listbox.curselection()[0])
        self.sol_current_term=select
        self.select_sol_all()
    def select_sol_source(self,w):
        select = int(self.sol_source_list.slistbox.listbox.curselection()[0])
        self.sol_current_term=select
        self.select_sol_all()
    def select_sol_solution(self,w):
        select = int(self.sol_solution_list.slistbox.listbox.curselection()[0])
        self.sol_current_term=select
        self.select_sol_all()
    def select_sol_point(self,w):
        select = int(self.sol_point_list.slistbox.listbox.curselection()[0])
        self.sol_current_term=select
        self.select_sol_all()
    def select_sol_bound_type(self,w):
        select = int(self.sol_bound_type_list.slistbox.listbox.curselection()[0])
        self.sol_current_term=select
        self.select_sol_all()
    def select_sol_all(self):
##        print "selecting term ", self.sol_current_term
        self.sol_diff_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.sol_source_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.sol_point_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.sol_bound_type_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.sol_solution_list.slistbox.listbox.selection_clear(0,last=Tix.END)
##        print "select current term ", self.sol_current_term
        if self.sol_current_term>-1:
            self.sol_diff_list.slistbox.listbox.selection_set(self.sol_current_term,last=self.sol_current_term)
            self.sol_source_list.slistbox.listbox.selection_set(self.sol_current_term,last=self.sol_current_term)
            self.sol_point_list.slistbox.listbox.selection_set(self.sol_current_term,last=self.sol_current_term)
            self.sol_bound_type_list.slistbox.listbox.selection_set(self.sol_current_term,last=self.sol_current_term)
            self.sol_solution_list.slistbox.listbox.selection_set(self.sol_current_term,last=self.sol_current_term)
            #set silent
            Enum = ["Dirichlet only","Neumann only,",
                          "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""] 
            sol_prob_list =  self.problems[self.sol_current_prob].prob_list
            self.sol_diff_list.set_silent(sol_prob_list[self.sol_current_term].DE)
            self.sol_source_list.set_silent(sol_prob_list[self.sol_current_term].sourceFunction)
            thing = [sol_prob_list[self.sol_current_term].point1,sol_prob_list[self.sol_current_term].point2]
            self.sol_point_list.set_silent(thing)
            self.sol_bound_type_list.set_silent(Enum[sol_prob_list[self.sol_current_term].boundaryTypeId])
            self.sol_solution_list.set_silent(sol_prob_list[self.sol_current_term].sln)
        else:
            self.sol_diff_list.set_silent("")
            self.sol_source_list.set_silent("")
            self.sol_point_list.set_silent("")
            self.sol_bound_type_list.set_silent("")   
            self.sol_solution_list.set_silent("") 

    def select_sol_problem(self,w):
##        print "selecting problem ", w
        #load self.diff_list, self.source_list, self.point1_list, self.point2_list, self.bound_type_list           
        #find index of problem
        if len(self.problems) >0:
            #delete the problems in the problem list
            self.sol_prob_list.slistbox.listbox.delete(0, END)
            #insert
            for i,p in enumerate(self.problems):
                self.sol_prob_list.insert(Tix.END, self.problems[i].name)
                
            self.sol_source_list.slistbox.listbox.delete(0, END)
            self.sol_diff_list.slistbox.listbox.delete(0, END)
            self.sol_point_list.slistbox.listbox.delete(0, END)
            self.sol_bound_type_list.slistbox.listbox.delete(0, END)
            self.sol_solution_list.slistbox.listbox.delete(0, END)
            self.sol_current_prob_name = w
            #find index of self.current_prob_name in self.problem_list
            #print map(lambda x:x.name, self.problems[self.current_prob].prob_list)
##            print "current_prob_name ", self.sol_current_prob_name, " len(self.problems) ",len(self.problems)
##            print map(lambda x: x.name, self.problems)
            self.sol_current_prob = map(lambda x: x.name, self.problems).index(self.sol_current_prob_name)
##            print "select problem ", w
            sol_prob_list =  self.problems[self.sol_current_prob].prob_list
            self.sol_prob_list.slistbox.listbox.selection_clear(0,last=Tix.END)
            self.sol_prob_list.slistbox.listbox.selection_set(self.sol_current_prob,last=self.sol_current_prob)
            #print self.current_prob, self.current_prob_name 
            #print  prob_list
            if len(sol_prob_list)>0:
                Enum = ["Dirichlet only","Neumann only","Arbitrary Mixed Dirichlet/Neumann",
                        "Random Mixed Dirichlet/Neumann",""]
                for i,p in enumerate(sol_prob_list):
                    #print type(p)
                    self.sol_diff_list.insert(Tix.END, p.DE)
                    self.sol_source_list.insert(Tix.END, p.sourceFunction)
                    self.sol_point_list.insert(Tix.END, [p.point1,p.point2])
                    self.sol_bound_type_list.insert(Tix.END, Enum[p.boundaryTypeId])
                    self.sol_solution_list.insert(Tix.END, p.sln)          
                self.sol_diff_list.set_silent(sol_prob_list[0].DE)
                self.sol_source_list.set_silent(sol_prob_list[0].sourceFunction)
                self.sol_point_list.set_silent([sol_prob_list[0].point1,sol_prob_list[0].point2])
                self.sol_bound_type_list.set_silent(Enum[sol_prob_list[0].boundaryTypeId])
                self.sol_solution_list.set_silent(sol_prob_list[0].sln)
                self.sol_current_term=0
                
    def sol_view_solution(self):
        if self.current_rep>-1:
            if self.representations[self.current_rep].representation_type == "implicit":
                rep = self.representations[self.current_rep]
                Solution.viewSolution(0, 0,rep.prob.boundDomain,  rep.param, \
                                      basis_type=rep.basis_function,representation=rep.representation_type)
            elif self.representations[self.current_rep].representation_type == "explicit":
                rep = self.representations[self.current_rep]
                Solution.viewSolution(0, 0,rep.prob.boundDomain,  rep.param, \
                                      basis_type=rep.basis_function,representation=rep.representation_type,\
                                      mcfall = rep.mcfall, mixed=rep.mcfall.mixed)
            
    def sol_view_network(self):
        if self.current_rep>-1:
            rep = self.representations[self.current_rep]
            Solution.viewSolution(0, 0,rep.prob.boundDomain,  rep.param, \
                                  basis_type=rep.basis_function,representation="implicit", title ="Network")   
    def sol_view_initial_error(self):
        if self.current_rep>-1 and self.representations[self.current_rep].representation_type == "explicit":
            rep = self.representations[self.current_rep]
            ##register the Energy Function with the Problem Information
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC,\
                                            mcfall = rep.mcfall)
##            print "parameters " , rep.param
            error = EnergyFunction.functions(rep.param)
            self.init_error.entry.delete(0,'end')
            self.init_error.entry.insert(0,str(math.log(error)/math.log(10)))
            
           # jacobian = EnergyFunction.jacobians(rep.param)
            #EnergyFunction.viewError()
            print "log error ", math.log(error)/math.log(10)
            #J= EnergyFunction.jacobians(rep.param, rep.probs.prob_list)
            #print "jacobian", J
            EnergyFunction.viewError()
        elif self.current_rep>-1 and self.representations[self.current_rep].representation_type == "implicit":
            rep = self.representations[self.current_rep]
            ##register the Energy Function with the Problem Information
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC)
##            print "parameters " , rep.param
            error = EnergyFunction.functions(rep.param)
            self.init_error.entry.delete(0,'end')
            self.init_error.entry.insert(0,str(math.log(error)/math.log(10)))
            
           # jacobian = EnergyFunction.jacobians(rep.param)
            #EnergyFunction.viewError()
            print "log error ", math.log(error)/math.log(10)
            #J= EnergyFunction.jacobians(rep.param, rep.probs.prob_list)
            #print "jacobian", J
            EnergyFunction.viewError()
        

    def performOptimisation(self):
        #load the energy function
        print "*************************************************"
        print "commencing optimisation"
        if self.current_rep>-1 and self.representations[self.current_rep].representation_type == "explicit":
            rep = self.representations[self.current_rep]
            ##register the Energy Function with the Problem Information
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC,\
                                            mcfall = rep.mcfall)
        elif self.current_rep>-1 and self.representations[self.current_rep].representation_type == "implicit":
            rep = self.representations[self.current_rep]
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC)

    ##        rep.param2 = copy.deepcopy(rep.param)
    ##        rep.param=Optimisation.optimise(EnergyFunction,rep.param)     
    ##        error = EnergyFunction.functions(rep.param)
        self.num_samples=int(self.select_samples['value'])
        print "num of samples " + str(self.num_samples)
        error_list = []
        error_parts_list=[]
        for i in range(self.num_samples):
            row = []
            print "sample: " +str(i)
            if self.sample_independent:
                param_aux = 0*self.generateRandomParameters()
            else:
                param_aux =copy.deepcopy(rep.param)
            if self.FirstTerm_option==True:     
                EnergyFunction.p_max = 1
                
            param_aux=Optimisation.optimise4(EnergyFunction,param_aux,self.num_runs,
                                             global_search_opt=self.global_search_opt)
            EnergyFunction.p_max = EnergyFunction.num_probs
            error = EnergyFunction.functions(param_aux)
            print "error " + str(error) + " rep.error " + str(rep.error)
            for p in range(EnergyFunction.num_probs):
                e = EnergyFunction.Errors[p]
                e_log = math.log(e)/math.log(10)
                row+=[e_log];
            error_parts_list+=[row]
            
            if error<rep.error:
                rep.error = error
                rep.param = copy.deepcopy(param_aux)
    ##        print error, error2
            error_list += [math.log(error)/math.log(10)];
            self.init_error.entry.delete(0,'end')
            self.init_error.entry.insert(0,str(math.log(rep.error)/math.log(10)))
            
        ##the mean and standard deviation
        error_std = std(error_list)
        error_mean = mean(error_list)
        self.results_list.listbox.delete(0, END)
        self.results_list.listbox.insert(0, "List log error :" + str(error_list))
        self.results_list.listbox.insert(0, "Num. samples :" + str(self.num_samples))
        self.results_list.listbox.insert(0, "Std. log error :" + str(error_std))
        self.results_list.listbox.insert(0, "Mean log error :" + str(error_mean))
        self.init_error.entry.delete(0,'end')
        self.init_error.entry.insert(0,str(math.log(rep.error)/math.log(10)))
        print "*************************************************"
        for p in range(EnergyFunction.num_probs):
            error_list = map(lambda x: x[p], error_parts_list)
            error_std = std(error_list)
            error_mean = mean(error_list)
            print "**********************************"
            print EnergyFunction.prob_list[p].DE
            print "Mean log error: " + str(error_mean)
            print "Std. log error: " + str(error_std)

    def analytical(self):
        print "*************************************************"
        print "commencing interpolation"
        #observe the error
        if self.current_rep>-1 and self.representations[self.current_rep].representation_type == "explicit":
            rep = self.representations[self.current_rep]
            ##register the Energy Function with the Problem Information
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC,\
                                            mcfall = rep.mcfall)
        elif self.current_rep>-1 and self.representations[self.current_rep].representation_type == "implicit":
            rep = self.representations[self.current_rep]
            DE = EnergyFunction.loadProblem(rep.probs, representation_type = rep.representation_type,\
                                            basis_type = rep.basis_function, \
                                            implicit_include_BC = rep.implicit_include_BC)

        ##check to ensure the problem is a target function mapping
        ##which means it contains a differential equation F[0][0] and source function
        rep = self.representations[self.current_rep]
        target_mapping = False
        j=0
        for p in rep.probs.prob_list:
            if p.DE == "F[0][0]":
                target_mapping = True
                break
            j+=1
        print "target_mapping: " +str(target_mapping)
        ##if so then procede then apply curve fitting
        self.num_samples=int(self.select_samples['value'])
        print "num samples " + str(self.num_samples)
        error_list = []
        removals_list = []
        for i in range(self.num_samples):

            if rep.representation_type == "implicit":
                if target_mapping:
                    approx = rep.probs.prob_list[j].SF2
                elif not target_mapping:
                    approx = approxFunction(rep.probs.prob_list[0].Xgrid, rep.probs.prob_list[0].Ygrid,
                                            rep.probs.prob_list[0].boundDomain, rep.probs.prob_list[0].d_Ohm_D, \
                                            rep.probs.prob_list[0].DC, 0, interpolation_type = self.interpolation_option["value"],
                                            xn_max=0,yn_max=0)
                    approx =approx.fns
                    A = []
                    for a in approx:
                        row=[]
                        for b in a:
                            row+=[b[0][0]]
                        A +=[row]
                    approx=A
                    
                    
                print "approx", approx
##                print "rep.probs.prob_list[0].boundDomain",rep.probs.prob_list[0].boundDomain
                (param,removals) = InterpolationMethod(rep.probs.prob_list[0].boundDomain,
                                                approx, rep.basis_function)
            elif rep.representation_type == "explicit":
                if target_mapping == False:
                    j=0
                approx = Solution.calculateNetwork2(0,0, rep.probs.prob_list[j].boundDomain,
                                               rep.probs.prob_list[j].sourceFunction, rep.mcfall)

                (param,removals) = InterpolationMethod(rep.probs.prob_list[j].clippedDomain,
                                                    approx, rep.basis_function)    
                #print param
            #first obtain boundary satisfying function using Dirichlet boundary
                
            ##if not then it is a differential equation of an unknown target mapping but
            ##we can use the Dirichlet boundary segments to
            ##form a starting solution and procede to use optimisation accordingly?
            error = EnergyFunction.functions(param)
            print "sample: " +str(i)
            print "error " + str(error)
            error_list += [math.log(error)/math.log(10)];
            removals_list += [removals]
            if error <rep.error:
                rep.param = param
                rep.error = error
            error_std = std(error_list)
            error_mean = mean(error_list)
            node_removals = median(removals_list)
            print "*************************************************"
            self.results_list.listbox.delete(0, END)
            self.results_list.listbox.insert(0, "List log error :" + str(error_list))
            self.results_list.listbox.insert(0, "Num. samples :" + str(self.num_samples))
            self.results_list.listbox.insert(0, "Median node removals :" + str(node_removals))
            self.results_list.listbox.insert(0, "Std. log error :" + str(error_std))
            self.results_list.listbox.insert(0, "Mean log error :" + str(error_mean))
            self.init_error.entry.delete(0,'end')
            self.init_error.entry.insert(0,str(math.log(rep.error)/math.log(10)))
        if self.current_rep>-1:
            self.select_sol_representation(self.representations[self.current_rep].name)
        
def RunMain(root):
    global test
    test = Test(root)

    test.build()
##    test.loop()
##    test.destroy()
    root.mainloop()  

        
if __name__ == '__main__':
    root = Tix.Tk()
    RunMain(root)

