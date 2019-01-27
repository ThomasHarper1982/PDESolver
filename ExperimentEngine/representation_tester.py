import os, os.path, sys, Tix
from Tkconstants import *
from ProblemInformation import *
from sympy import *

TCL_DONT_WAIT       = 1<<1
TCL_WINDOW_EVENTS   = 1<<2
TCL_FILE_EVENTS     = 1<<3
TCL_TIMER_EVENTS    = 1<<4
TCL_IDLE_EVENTS     = 1<<5
TCL_ALL_EVENTS      = 0

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
        self.problem_list = []
        self.DE_list = []
        self.source_func = []
        self.point1_list = []
        self.point2_list = []
        self.view_mode_opt = "source"
        self.func_assign_opt = "solution"
##        self.max_prob = 0
##        self.current_prob_name = ""
        self.current_prob= -1
##        self.max_prob_term = []
        self.unique=0
##        self.current_prob_term=-1
        self.current_term =-1
        self.func ="y**2*sin(pi*x)"
        self.DE = "F[0][0]"
        self.boundType = 0
        self.num_points = 9
        self.point1 = [0,0]
        self.point2 = [1,1]
        self.x_order = 0
        self.y_order = 0
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
        print sys.path
        
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
        w.add('opt', label='Optimisation', underline=0,
              createcmd=lambda w=w, name='opt': self.ProblemParameters(w, name))
        w.add('exp', label='Experiment', underline=0,
              createcmd=lambda w=w, name='exp': self.ProblemParameters(w, name))
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
        z.wm_title('Tix Widget Demonstration')
        z.geometry('790x590+10+10')
        test.balloon = Tix.Balloon(root)
        frame1 = self.MkMainMenu()
        frame2 = self.MainNotebook()
        frame3 = self.MkMainStatus()
        frame1.pack(side=TOP, fill=X)
        #frame3.pack(side=BOTTOM, fill=X)
        frame2.pack(side=TOP, expand=1, fill=BOTH, padx=4, pady=4)
        test.balloon['statusbar'] = test.statusbar
        z.wm_protocol("WM_DELETE_WINDOW", lambda self=self: self.quitcmd())

    def quitcmd (self):
        """Quit our mainloop. It is up to you to call root.destroy() after."""
        self.exit = 0

    def loop(self):
        import tkMessageBox, traceback
        while self.exit < 0:
            try:
                self.root.tk.dooneevent(TCL_ALL_EVENTS)
            except SystemExit:
                #print 'Exit'
                self.exit = 1
                break
            except KeyboardInterrupt:
                if tkMessageBox.askquestion ('Interrupt', 'Really Quit?') == 'yes':
                    # self.tk.eval('exit')
                    return
                else:
                    pass
                continue
            except:
                t, v, tb = sys.exc_info()
                text = ""
                for line in traceback.format_exception(t,v,tb):
                    text += line + '\n'
                try: tkMessageBox.showerror ('Error', text)
                except: pass
                tkinspect_quit (1)

    def destroy (self):
        self.root.destroy()

####################Page 1: Problem Parameters##########################
##Widget hierarchy
    def ProblemParameters(self, nb, name):
        print "problem"
        w = nb.page(name)
        prefix = Tix.OptionName(w)
        if not prefix:
            prefix = ''
        w.option_add('*' + prefix + '*TixLabelFrame*label.padX', 4)

        func = Tix.LabelFrame(w, label='functions')
        bound = Tix.LabelFrame(w, label='rectangular boundary')
        de = Tix.LabelFrame(w, label='differential equation')
        bc = Tix.LabelFrame(w, label='boundary conditions')
        monitor = Tix.LabelFrame(w, label='Problem set')

        self.ProbMonitor(monitor.frame)    
        self.ProbFunc(func.frame)
        self.ProbBound(bound.frame)
        self.ProbDiffEqu(de.frame)
        self.ProbBoundCond(bc.frame)

        monitor.form(left=0, right=-1, top=0)
        func.form(left=0, right='%50',top=monitor)
        de.form(left=func, right=-1, top=monitor)
        bound.form(left=0, right='&'+str(func), top=func)
        bc.form(left=func, right=-1, top=de)
        
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
        
    def ProbFunc(self, w):
        self.f_list = Tix.ComboBox(w, label="Function List: ", dropdown=0,
            command=lambda w=w: self.selectFunc(w), editable=1, variable=self.func,
            options='listbox.height 3 label.padY 5 label.width 10 label.anchor ne')
        self.f_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.f_list.insert(Tix.END, 'y**2*sin(pi*x)')
        self.f_list.insert(Tix.END, '((x-0.5)**2 + (y-0.5)**2 +1)**(1/2.0)')
        self.f_list.insert(Tix.END, 'cos(x*2*pi)*sin(y*2*pi)')
        self.f_list.insert(Tix.END, 'exp(-x)*(x+y**3)')
        self.f_list.insert(Tix.END, 'ln(1+(x-0.5)**2 +(y-0.5)**2)')
        self.f_list.set_silent('y**2*sin(pi*x)')
        self.f_list.pack(fill=Tix.X, padx=5, pady=3)
        x_order = Tix.DoubleVar()
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

        xn.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        yn.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('diff', text='Differentiate', underline=0, width=9,
                command=self.differentiate_fn)
        box.pack(side=Tix.BOTTOM,padx=5, pady=3, fill=Tix.X)
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
        self.diff_e = Tix.ComboBox(w, label="Differential Equation List: ", dropdown=0,
            command=lambda w=w: self.select_DE(w), editable=1, variable=self.DE,
            options='listbox.height 5 label.padY 5 label.width 25 label.anchor ne')
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
        self.diff_list = Tix.ComboBox(w, label="DEs: ", dropdown=0,
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
##        point1_label = Tix.Label(w, text='Point 1:')
##        point1_val = Tix.Label(w, text='')
##        point2_label = Tix.Label(w, text='Point 2:')
##        point2_val = Tix.Label(w, text='')
##        bc_label = Tix.Label(w, text='Boundary Condition Type :')
##        bc_val = Tix.Label(w, text='')

##        bc_type.pack(side=Tix.TOP, anchor=Tix.W)
    ##
##        point1_label.pack(side=Tix.LEFT, padx=3, pady=3, fill=Tix.BOTH)
##        point1_val.pack(side=Tix.LEFT, padx=3, pady=3, fill=Tix.BOTH)
##        point2_label.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
##        point2_val.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
##     
##        bc_label.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
##        bc_val.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
    ##def MKMonitorButtons(w):
        
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
        print self.DE

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
        print "selecting term ", self.current_term
        self.diff_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.source_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.point1_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.point2_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.bound_type_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        self.solution_list.slistbox.listbox.selection_clear(0,last=Tix.END)
        print "select current term ", self.current_term
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
        print "selecting problem ", w
        #load self.diff_list, self.source_list, self.point1_list, self.point2_list, self.bound_type_list           
        #find index of problem
        if len(self.problems) >0:
            self.source_list.slistbox.listbox.delete(0, END)
            self.diff_list.slistbox.listbox.delete(0, END)
            self.point1_list.slistbox.listbox.delete(0, END)
            self.point2_list.slistbox.listbox.delete(0, END)
            self.bound_type_list.slistbox.listbox.delete(0, END)
            self.current_prob_name = w
            #find index of self.current_prob_name in self.problem_list
            #print map(lambda x:x.name, self.problems[self.current_prob].prob_list)
            print "current_prob_name ", self.current_prob_name, " len(self.problems) ",len(self.problems)
            print map(lambda x: x.name, self.problems)
            self.current_prob = map(lambda x: x.name, self.problems).index(self.current_prob_name)
            print "select problem ", w
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
        print "new problem ",self.current_prob
            #print self.current_prob,self.max_prob,self.current_prob_term,self.max_prob_term[self.current_prob]  
##        print self.problems[self.current_prob].prob_list
##            print self.current_prob,self.max_prob,self.current_prob_term,self.max_prob_term[self.current_prob]
    def insertDE(self):
        if  len(self.problems) >0:
    ##        print "insert"
            Enum = ["Dirichlet only","Neumann only",
                      "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""] 
            #modify existing instance
            #if self.max_prob_term[self.current_prob]> 0:
            #self.max_prob_term[self.current_prob]+=1
            
            prob =self.problems[self.current_prob].insert()
            
            #load problem parameters
            #prob =  self.problems[self.current_prob].prob_list[self.current_term]
            self.point1 = eval(self.point1_lab.entry.get())
            self.point2 = eval(self.point2_lab.entry.get())
            self.num_points = eval(self.num_points_lab.entry.get())
            self.DE = self.diff_e.entry.get()
            self.func = self.f_list.entry.get()
            self.boundType = Enum.index(self.bc.entry.get())

           # print func_assign_option.state
            #self.func_assign_op = func_assign_option.state
            self.func_assign_opt= self.func_assign_option["value"]
  
            prob.setDomain(point1 = self.point1,point2=self.point2,numpoints=self.num_points)
            prob.setBoundaryPoints(0)
            prob.setDifferentialEquation(self.DE)
            prob.setDirichletNeumannAngles(self.boundType,numAngles=4,nstate=1)
            prob.setBoundarySegments()
            prob.setBoundaryValues(self.func)
            if self.func_assign_opt == "solution":
                print "solution"
                prob.deriveSourceFunction(self.func)
            elif self.func_assign_opt == "source":
                print "source"
                prob.setSourceFunction(self.func)
            else:
                print "no state"
                
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
            print "current term, ",  self.current_term
            self.select_all()                
    ##            print self.max_prob_term
    ##            print self.current_prob,self.max_prob,self.current_prob_term,self.max_prob_term[self.current_prob]
    ##        print self.max_prob_term
    ##        print self.problems[self.current_prob].prob_list
    def viewFunction(self):
        self.view_mode_opt= self.view_mode_option["value"]
        print self.view_mode_opt
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
                print "problem, no terms"
                print "num probs ", len(self.problems)
                self.problems = self.problems[0:self.current_prob]+self.problems[self.current_prob+1:]
                print "num probs ", len(self.problems)
                self.prob_list.slistbox.listbox.delete(self.current_prob, self.current_prob)
                self.prob_list.set_silent("")
                self.current_prob=len(self.problems)-1
                if self.current_prob>-1:
                    print "name ", self.problems[self.current_prob].name
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
        print self.x_order
        
    def select_Y_order(self,w):
        self.y_order = w
        print self.y_order
        #w.pack(side=Tix.TOP, fill=Tix.BOTH, expand=0)
        
    def selectFunc(self, w):
     #   print "here"
        #print self.func
        self.func = w
        print self.func

    def differentiate_fn(self, event=None):
        # tixDemo:Status "Year = %s" % demo_year.get()
        x=Symbol('x')
        y=Symbol('y')
        print self.func, self.x_order,self.y_order
        fn=diff(eval(self.func),x,self.x_order)
        fn=diff(fn,y,self.y_order)
        self.func = str(fn)
        self.func=self.func.replace("1/2", "1.0/2")
        self.f_list.set_silent(self.func)
        print self.func
    def select_bound(self, w, event=None):
        # tixDemo:Status "Year = %s" % demo_year.get()
        Enum = ["Dirichlet only","Neumann only",
                  "Arbitrary Mixed Dirichlet/Neumann","Random Mixed Dirichlet/Neumann",""]
        self.boundType = Enum.index(w)
        print self.boundType
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
        print "solution"
        w = nb.page(name)
        prefix = Tix.OptionName(w)
        if not prefix:
            prefix = ''
        w.option_add('*' + prefix + '*TixLabelFrame*label.padX', 4)

        rep_monitor = Tix.LabelFrame(w, label='representation workbench')
##        de = Tix.LabelFrame(w, label='Current Differential Equations')
##        bound = Tix.LabelFrame(w, label='Static Boundary Information')
        rep_prob = Tix.LabelFrame(w, label='Problem options')
        rep_choices = Tix.LabelFrame(w, label='representation options')
        
        self.SolutionRepresentationMonitor(rep_monitor.frame)
        self.SolutionProb(rep_prob.frame)
        self.RepresentationChoicesSelection(rep_choices.frame)    # a
        
        rep_monitor.form(left=0, right=-1, top=0)
        rep_prob.form(left=0, right=-1, top=rep_monitor)
        rep_choices.form(left=0, right=-1, top=rep_prob)

        
    def SolutionRepresentationMonitor(self, w):
        self.rep_list = Tix.ComboBox(w, label="Representation: ", dropdown=0,
            command=self.select_representation, editable=0, 
            options='listbox.height 3 label.padY 2 label.width 15 label.anchor ne')
        self.rep_list.pack(side=Tix.TOP, anchor=Tix.W)
        self.rep_info_list = Tix.ComboBox(w, label="Representation info: ", dropdown=0,
            command=self.select_rep_info_list, editable=0,
            options='listbox.height 3 label.padY 2 label.width 15 label.anchor ne')
        self.rep_problem_list = Tix.ComboBox(w, label="Problem List: ", dropdown=0,
            command=self.select_rep_prob_list, editable=0,
            options='listbox.height 3 label.padY 2 label.width 10 label.anchor ne')
        self.rep_problem_info_list = Tix.ComboBox(w, label="Problem info list: ", dropdown=0,
            command=self.select_rep_problem_info_list, editable=0,
            options='listbox.height 3 label.padY 2 label.width 15 label.anchor ne')
        
        self.rep_list.pack(side=Tix.LEFT, anchor=Tix.W) 
        self.rep_info_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.rep_problem_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
        self.rep_problem_info_list.pack(side=Tix.LEFT, padx=5, pady=3, fill=Tix.X)
    def SolutionProb(self, w):
        de = Tix.LabelFrame(w, label='Current Problems')
        bound = Tix.LabelFrame(w, label='Boundary Information')

        self.ProbMonitorDiffEqu(de.frame)
        self.ProbMonitorBoundary(bound.frame)    
        
        de.pack(side=Tix.TOP, anchor=Tix.W)
        bound.pack(side=Tix.TOP, anchor=Tix.W)
    def RepresentationChoicesSelection(self,w):
        self.basis_function_option = Tix.Select(w, label='Basis Function:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.basis_function_option.add('sigmoid', text='Sigmoid') #command=self.setSolAssign())
        self.basis_function_option.add('sinusoid', text='Sinusoid')
        self.basis_function_option.add('radial_basis', text='Radial')
        
        self.bound_representation_option = Tix.Select(w, label='Boundary Representation Type:', allowzero=1, radio=1,orientation=Tix.VERTICAL)
        self.bound_representation_option.add('explicit', text='Explicit') #command=self.setSolAssign())
        self.bound_representation_option.add('implicit', text='Implicit')
##    def RepresentationProblemFilterSelection(self,w):
        self.basis_function_option.pack(side=Tix.LEFT, anchor=Tix.W)
        self.bound_representation_option.pack(side=Tix.LEFT, anchor=Tix.W)
        
    def RepresentationControl(self, w):
        # 4 buttons
        #new
        #insert
        #delete
        #view
        box = Tix.ButtonBox(w, orientation=Tix.HORIZONTAL)
        box.add('new', text='New', underline=0, width=5,
                command=lambda w=w: self.newRep())
        box.add('delete', text='Delete', underline=0, width=5,
                command=lambda w=w: self.deleteRep())   
        box.add('view', text='View', underline=0, width=5,
                command=lambda w=w: self.viewFunction())

##        box.add('boundary', text='Bound', underline=0, width=5,
##                command=lambda w=w: self.viewBoundary())
        box.pack(side=Tix.BOTTOM, fill=Tix.X)
        box.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)
        
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
    def select_representation(self,w):
        pass

    def select_rep_info_list(self,w):
        pass 

    def select_rep_prob_list(self,w):
        pass

    def select_rep_problem_info_list(self,w):
        pass


##static functions
def RunMain(root):
    global test

    test = Test(root)

    test.build()
    test.loop()
    test.destroy()
    

        
if __name__ == '__main__':
    root = Tix.Tk()
    RunMain(root)
