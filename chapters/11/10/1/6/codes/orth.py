#Code by GVV Sharma
#September 12, 2023
#Revised July 21, 2024
#released under GNU GPL
#Orthogonality


import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts
import numpy as np
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

#Given points
A=np.array(([4,4])).reshape(-1,1)
B=np.array(([3,5])).reshape(-1,1)
C=np.array(([-1,-1])).reshape(-1,1)
if (A-B).T@(B-C)==0:
   print("AB is perpendicular to BC and hence the triangle is right angled")
elif (B-C).T@(C-A)==0:
   print("BC  is perpendicular to CA and hence the triangle is right angled")
elif (C-A).T@(A-B)==0:
   print("CA is perpendicular to AB and hence the triangle is right angled")
#Generating Lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
X_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(X_CA[0,:],X_CA[1,:],label='$CA$')

colors = np.arange(1,4)
#Labeling the coordinates
tri_coords = np.block([[A,B,C]])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    #plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.0f}, {tri_coords[1,i]:.0f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(20,0), # distance from text to points (x,y)
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
plt.savefig('chapters/11/10/1/6/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/11/10/1/6/figs/fig.pdf"))
#else
#plt.show()
