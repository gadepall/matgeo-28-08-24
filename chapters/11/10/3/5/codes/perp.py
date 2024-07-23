#Code by GVV Sharma
#July 22, 2024
#released under GNU GPL
#Line 


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

#Direction vector
n = np.array(([4, 3])).reshape(-1,1) 
m = np.array(([-3, 4])).reshape(-1,1) 
c,d =12,4

#Points
x1 = (d*LA.norm(n)+c)/(n.T@e1)
x2 = (-d*LA.norm(n)+c)/(n.T@e1)

P = x1*e1
Q = x2*e1

#Foot of the perpendicular
R1 = perp_foot(n,c,P)
R2 = perp_foot(n,c,Q)


#Generating Lines
k1 = -2
k2 = 2
x_Q = line_dir_pt(m,R1,k1,k2)
x_PR1 = line_gen(P,R1)
x_QR2 = line_gen(Q,R2)

#Plotting all lines
plt.plot(x_Q[0,:],x_Q[1,:],label='$line$')
plt.plot(x_PR1[0,:],x_PR1[1,:],label='$PR1$')
plt.plot(x_QR2[0,:],x_QR2[1,:],label='$QR2$')

colors = np.arange(1,3)
#Labeling the coordinates
tri_coords = np.block([P,Q])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['P','Q']
for i, txt in enumerate(vert_labels):
    #plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.0f}, {tri_coords[1,i]:.0f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
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
plt.savefig('chapters/11/10/3/5/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/11/10/3/5/figs/fig.pdf"))
#else
#plt.show()
