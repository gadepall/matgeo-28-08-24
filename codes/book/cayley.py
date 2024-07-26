#Code by GVV Sharma
#July 26, 2024
#released under GNU GPL
#Cayley Hamilton Theorem

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts
import numpy as np
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
A = np.array(([1, 0, 2],[0, 2, 1],[2, 0, 3]))
#Characteristic polynomial of A
f = np.poly(A)
#Evaluate matrix polynomial
result = polyvalm(f, A)
print(result)

