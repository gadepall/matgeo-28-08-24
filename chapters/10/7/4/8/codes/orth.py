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
A = np.array(([-1,-1])).reshape(-1,1) 
B = np.array(([-1,4])).reshape(-1,1) 
C = np.array(([5,4])).reshape(-1,1) 
D = np.array(([5,-1])).reshape(-1,1) 

#mid points
P = (A+B)/2
Q = (B+C)/2
R = (C+D)/2
S = (D+A)/2

#We know that the figure formed by joining mid points of a quadrilateral is a parallelogram.
#To establish, if it is a rectangle, we will compute the dot product of any 2 adjacent sides
#if the dot product is zero, then it is rectangle

if ( (Q-P).T@(R-Q) == 0):   #Check, if any 2 adjacent sides are orthogonal
    if ((R-P).T@(S-Q) == 0): #If diagonals are orthogonal, then it is square
        print("PQRS is a Square")
    else:                      #If diagonals are not orthogonal, then it is rectangle
        print("PQRS is a rectangle")
else:                            
    if ((R-P).T@(S-Q) == 0):   #if diagonals are orthogonal, then it is rhombus
        print("PQRS is a Rhombus")
    else:
        print("PQRS is a parallelogram") # Else it is parallelogram

#Generating Lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)

x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$AD$')

plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$')
plt.plot(x_RS[0,:],x_RS[1,:],label='$RS$')
plt.plot(x_SP[0,:],x_SP[1,:],label='$PS$')

colors = np.arange(1,9)
#Labeling the coordinates
tri_coords = np.block([[A,B,C,D,P,Q,R,S]])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['A','B','C','D','P','Q','R','S']
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
'''
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/10/7/4/8/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/10/7/4/8/figs/fig.pdf"))
#else
#plt.show()
