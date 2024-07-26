#Code by Parv Chandola
#May 11, 2024
#Revised July 26, 2024
#by GVV Sharma
#released under GNU GPL
#Circle equation

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys               #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts

#local imports
from conics.funcs import circ_gen
from line.funcs import *

#if using termux
import subprocess
import shlex

C= np.array([2,-3]).reshape(-1,1) 
B= np.array([1,4]).reshape(-1,1) 

#line parameters
A = 2*C-B

#Centre and radius
r = LA.norm(B-C)
print(C,r)

#generating circle
x_circ= circ_gen(C,r)


#Generating Lines
x_AB = line_gen(A,B)

#Plotting all lines and circles
plt.plot(x_AB[0,:],x_AB[1,:],label='$(4 \quad 1)\mathbf{x}=16$')
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

colors = np.arange(1,4)
#Labeling the coordinates
tri_coords = np.block([A,B,C])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    #plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.0f}, {tri_coords[1,i]:.0f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-20,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
'''
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
'''
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/10/7/2/7/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/10/7/2/7/figs/fig.pdf"))
#else
#plt.show()
