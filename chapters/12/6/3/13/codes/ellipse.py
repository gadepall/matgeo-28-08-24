#Program to plot the tangent to a hyperbola
#Code by GVV Sharma
#August 8, 2020
#Revised August 16, 2024
#Revised August 21, 2024
#Revised August 22, 2024

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
y = np.linspace(-2,2,num)

#hyperbola parameters, first eigenvalue should be negative
V = np.array(([16,0],[0,9]))
u = np.zeros((2,1))
#u = -0.5*np.array(([0,1])).reshape(-1,1)
f = -16*9
n0,c,F,O,lam,P,e = conic_param(V,u,f)
#print(O)
ab = ellipse_param(V,u,f)
#Generating the Standard ellipse
xStandard= ellipse_gen(ab[0],ab[1])
ParamMatrix = np.diag(ab)
P = rotmat(np.pi/2)

#Tangent
'''
q = np.zeros((2,1))
q[0][0] = 10
q[1][0] = (q[0][0]-1)/(q[0][0]-2)
'''
#n = np.array(([1,1])).reshape(-1,1)
n = e2
q = conic_contact(V,u,f,n)
c = (n.T@q).flatten()
n1 = e1
q1 = conic_contact(V,u,f,n1)
c1 = (n1.T@q1).flatten()
#n = V@q+u
#c = n.T@q
#print(n,c)
#Directrix

#Affine conic generation
Of = O.flatten()
#Generating lines
k1 = -3
k2 = 3
x_A = line_norm(n,c[0],k1,k2)
x_B = line_norm(n,c[1],k1,k2)
x_C = line_norm(n1,c1[0],k1,k2)
x_D = line_norm(n1,c1[1],k1,k2)
'''
x_A = P@line_norm(n,c[0],k1,k2)+Of[:,np.newaxis]#directrix
x_B = P@line_norm(n,cl[0],k1,k2)+Of[:,np.newaxis]#latus rectum
x_C = P@line_norm(n,c[1],k1,k2)+Of[:,np.newaxis]#directrix
x_D = P@line_norm(n,cl[1],k1,k2)+Of[:,np.newaxis]#latus rectum

xStandardHyperLeft = np.block([[-x],[y]])
xStandardHyperRight= np.block([[x],[y]])
'''



#Generating the actual ellipse
#xActual = P@xStandard + Of[:,np.newaxis]
#xActualHyperLeft = P@ParamMatrix@xStandardHyperLeft+Of[:,np.newaxis]
#xActualHyperRight = P@ParamMatrix@xStandardHyperRight+Of[:,np.newaxis]


#plotting
#plt.plot(xActualHyperLeft[0,:],xActualHyperLeft[1,:],label='Hyperbola',color='r')
#plt.plot(xActualHyperRight[0,:],xActualHyperRight[1,:],color='r')
#plt.plot(xActual[0,:],xActual[1,:],label='Ellipse')
plt.plot(xStandard[0,:],xStandard[1,:],label='Ellipse')
plt.plot(x_A[0,:],x_A[1,:],label='X Tangent',color='r')
plt.plot(x_B[0,:],x_B[1,:],color='r')
plt.plot(x_C[0,:],x_C[1,:],color='g')
plt.plot(x_D[0,:],x_D[1,:],label='Y Tangent',color='g')
#
colors = np.arange(1,4)
#Labeling the coordinates
tri_coords = np.block([O,q])
#tri_coords = np.block([O,F])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
#vert_labels = ['$\mathbf{O}$']
vert_labels = ['$\mathbf{O}$','$\mathbf{q}_1$','$\mathbf{q}_2$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
#    plt.annotate(f'{txt}\n({tri_coords[0,i]:.2f}, {tri_coords[1,i]:.2f})',
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(10,-10), # distance from text to points (x,y)
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
plt.savefig('chapters/12/6/3/13/figs/fig.pdf')
subprocess.run(shlex.split("termux-open chapters/12/6/3/13/figs/fig.pdf"))
#else
#plt.show()
