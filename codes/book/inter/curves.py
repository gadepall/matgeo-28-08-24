#Program to plot  the tangent of a parabola
#Code by GVV Sharma
#Released under GNU GPL
#August 21, 2024

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/github/matgeo/codes/CoordGeo')        #path to my scripts


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
num= 100
#y = np.linspace(-2.5,2.5,num)

#Circle parameters
u_1 = -e1
f_1 = 0
O,r = circ_param(u_1,f_1)
x_A = circ_gen_num(O,r,num)
u = np.array(([0,0])).reshape(-1,1)
f = -1
O,r = circ_param(u,f)
print(r)
x_B = circ_gen_num(O,r,num)

#Chord parameters
n = u - u_1
c = (f_1-f)/2
m,h = param_norm(n,c)
A,B = chord(np.eye(2),u,f,m,h)
print(A,B)

plt.plot(x_A[0,:],x_A[1,:],label='Circle 1')
plt.plot(x_B[0,:],x_B[1,:],label='Circle 2')
plt.fill_between(x_B[0],x_B[1],where= (0.5 <= x_B[0])&(x_B[0] <= 1 ), color = 'cyan')
plt.fill_between(x_A[0],x_A[1],where= (0 <= x_A[0])&(x_A[0] <= 0.539 ), color = 'cyan',label='Intersection')
colors = np.arange(1,3)
#Labeling the coordinates
tri_coords = np.block([A,B])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['$\mathbf{A}$','$\mathbf{B}$']
#vert_labels = ['$\mathbf{D}$','$\mathbf{E}$']
for i, txt in enumerate(vert_labels):
#    plt.annotate(txt, # this is the text
    plt.annotate(f'{txt}\n({tri_coords[0,i]:.2f}, {tri_coords[1,i]:.2f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

'''
#
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xlabel('$x$')
plt.ylabel('$y$')
'''
#plt.legend(loc='best')
plt.legend(loc='lower right')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/12/8/2/2/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/12/8/2/2/figs/fig.pdf"))
#else
#plt.show()
