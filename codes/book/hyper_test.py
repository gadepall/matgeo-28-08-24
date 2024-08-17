#Program to plot an ellipse 
#Code by GVV Sharma
#August 8, 2020
#Revised July 31, 2024
#Revised August 16, 2024

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
len = 100
y = np.linspace(-2,2,len)

#Conic parameters
'''
e = 4/3
P1 =  7*e1
P2 =  -7*e1
'''
P1 =  3*e2
P2 =  -3*e2
V = np.array(([-16/9,0],[0,1]))
u = np.array(([0,0])).reshape(-1,1)
f = 16
n,c,F,O,lam,P,e = conic_param(V,u,f)
ab = ellipse_param(V,u,f)

#Generating the Standard Hyperbola
x = hyper_gen(y)
ParamMatrix = np.diag(ab)
print(e)


#Affine conic generation
P = rotmat(np.pi/2)
Of = O.flatten()

xStandardHyperLeft = np.block([[-x],[y]])
xStandardHyperRight= np.block([[x],[y]])


#Generating the eigen hyperbola
xeigenHyperLeft = ParamMatrix@xStandardHyperLeft
xeigenHyperRight = ParamMatrix@xStandardHyperRight

#Generating the actual hyperbola
xActualHyperLeft = P@ParamMatrix@xStandardHyperLeft+Of[:,np.newaxis]
xActualHyperRight = P@ParamMatrix@xStandardHyperRight+Of[:,np.newaxis]


#plotting
#Plotting the actual hyperbola
plt.plot(xActualHyperLeft[0,:],xActualHyperLeft[1,:],label='hyperbola',color='r')
plt.plot(xActualHyperRight[0,:],xActualHyperRight[1,:],color='r')
#
colors = np.arange(1,6)
#Labeling the coordinates
tri_coords = np.block([O,F,P1,P2])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['$\mathbf{O}$','$\mathbf{F}_1$','$\mathbf{F}_2$', '$\mathbf{P}_1$','$\mathbf{P}_2$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
#    plt.annotate(f'{txt}\n({tri_coords[0,i]:.2f}, {tri_coords[1,i]:.2f})',
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
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
'''
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/11/11/4/9/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/11/11/4/9/figs/fig.pdf"))
#else
#plt.show()
