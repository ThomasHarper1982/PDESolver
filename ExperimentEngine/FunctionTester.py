# interface design, let the user choose options from drop down boxes
# {problem 1-10}; {sigmoid, RBF, Sine Wave}; {implicit, explicit}, {5,10,15,20 nodes}

import Tkinter
import ScrolledText
import tkFileDialog
import tkSimpleDialog
import random

class FunctionTester:
    
     def __init__(self,parent):
          self.parent = parent
          self.textWidget = ScrolledText.ScrolledText(parent, width=80, height=50)
          self.textWidget.pack()
          self.menuBar = Tkinter.Menu(parent, tearoff=0)
          
          
          
          
    
root =Tkinter.Tk()
testApp = FunctionTester(root)
root.mainloop()
