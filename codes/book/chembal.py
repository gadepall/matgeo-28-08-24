#Code by GVV Sharma
#Jan 23, 2021
#Revised July 26, 2024
#released under GNU GPL
#Balance chemical equations

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts
import numpy as np
import sympy
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#local imports
from line.funcs import *
from triangle.funcs import *
from matrix.funcs import *
from conics.funcs import circ_gen


#if using termux
import subprocess
import shlex
#end if

#Given points
A = np.array(([1, 2, 0, -2],[1, 0, -2, 0],[3, 2, -6, -1],[0, 1, -1, 0]))
print(sympy.Matrix(A).rref())
