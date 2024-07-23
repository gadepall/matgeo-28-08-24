#Code by GVV Sharma
#July 22, 2024
#released under GNU GPL
#Affine Transformation


import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts
import numpy as np
import mpmath as mp
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen


#if using termux
import subprocess
import shlex
#end if

I = np.eye(2)
e1 = I[:,[0]]
e2 = I[:,[1]]
a = 2
#Given vertices of the square 
A = np.sqrt(a)*e1
B = a*e2
C = -a*e2
D = -np.sqrt(a)*e1

#Generating Lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_CD = line_gen(C,D)
x_DB = line_gen(D,B)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$distance(AB)$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$distance(BC)$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$distance(CA)$')
plt.plot(x_DB[0,:],x_DB[1,:],label='$distance(DB)$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$distance(CD)$')

colors = np.arange(1,5)
#Labeling the coordinates
tri_coords = np.block([[A,B,C,D]])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['A','B','C','$A^\prime$']
for i, txt in enumerate(vert_labels):
#    plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.2f}, {tri_coords[1,i]:.2f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(30,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

# use set_position
ax = plt.gca()
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
'''
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
'''
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/11/10/1/2/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/11/10/1/2/figs/fig.pdf"))
#else
#plt.show()
