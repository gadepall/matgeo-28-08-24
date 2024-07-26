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

x1 = np.array([6,-6]).reshape(-1,1) 
x2 = np.array([3,-7]).reshape(-1,1) 
x3 = np.array([3,3]).reshape(-1,1) 


#Entering equations inmatrix form
A = np.block([[2*x1,2*x2,2*x3],[1,1,1]]).T
b = -np.array([LA.norm(x1)**2,LA.norm(x2)**2,LA.norm(x3)**2]).reshape(-1,1) 
x = LA.solve(A,b)

#Centre and radius
u = x[:2]
O = -u
f = x[2][0]
r = np.sqrt(LA.norm(u)**2-f)
print(O,r)

#generating circle
x_circ= circ_gen(O,r)


#Plotting all lines and circles
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

colors = np.arange(1,5)
#Labeling the coordinates
tri_coords = np.block([x1,x2,x3,O])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['$\mathbf{x}_1$','$\mathbf{x}_2$','$\mathbf{x}_3$','$\mathbf{O}$']
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
'''
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/10/7/4/3/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/10/7/4/3/figs/fig.pdf"))
#else
#plt.show()
