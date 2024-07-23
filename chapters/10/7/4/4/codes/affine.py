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
#Given vertices of the square 
A = np.array([-1,2]).reshape(-1,1)
C = np.array([3,2]).reshape(-1,1)

c = A#translation vector
m = C-A #direction vector
slope = m[1]/m[0]

#angle made with the x axis
phi=  np.arctan(slope[0])
theta = phi-np.pi/4 
P = rotmat(theta)#rotation matrix

#Standard square
sts = LA.norm(C-A)/np.sqrt(2)*np.block([np.zeros((2,1)),I[:,[0]],np.ones((2,1)),I[:,[1]]])
sts = P@sts
sts = sts+np.block([A,A,A,A])
#remaining vertices using affine transform
#vert=LA.norm(C-A)*P/np.sqrt(2)+np.block([A,A])

#print(np.zeros((2,1)))
print(sts)

A = sts[:,[0]]
B = sts[:,[1]]
C = sts[:,[2]]
D = sts[:,[3]]


#Generating Lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$distance(AB)$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$distance(BC)$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$distance(CD)$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$distance(DA)$')

colors = np.arange(1,5)
#Labeling the coordinates
#tri_coords = np.block([[A,B,C,P,Q,R]])
tri_coords = sts
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
#    plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.2f}, {tri_coords[1,i]:.2f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(25,15), # distance from text to points (x,y)
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
plt.savefig('chapters/10/7/4/4/figs/fig2.pdf')
subprocess.run(shlex.split("termux-open chapters/10/7/4/4/figs/fig2.pdf"))
#else
#plt.show()
