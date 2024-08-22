#Program to plot  the tangent of a parabola
#Code by GVV Sharma
#Released under GNU GPL
#August 19, 2024

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

V = np.array(([4,0],[0,9]))
u = np.array(([0,0])).reshape(-1,1)
f = -4*9 
n = np.array(([2,3])).reshape(-1,1)
c = 2*3 
m,h = param_norm(n,c)
#A,B = chord(V,u,f,m,h)
q = chord(V,u,f,m,h)
A = q[:,0]
B = q[:,1]
print(q)
#print(h,m)
#print(A,B)

'''
#conic parameters
V = np.array(([0,0],[0,1]))
#u = -8*e1
P = rotmat(-np.pi/2)
u = -625/12*P@e1
print( -625/12*P@e2)
f = 0

n,c,F,O,lam,P,e = conic_param(V,u,f)
#print(n,c,F)
#print(lam,P)

#Eigenvalues and eigenvectors

#flen = parab_param(lam,P,u)
print(flen,e)
flen = 1
'''

#Standard parabola generation
ab = ellipse_param(V,u,f)
xStandard= ellipse_gen_num(ab[0],ab[1],num)
#x = circ_gen(u,2)
#x = parab_gen(y,flen)

#Cable
x_A = line_gen_num(A,B,num)
#x_B = line_gen_num(C,D,num)


#Directrix
k1 = -8
k2 = 8

#Latus rectum
#cl = (n.T@F).flatten()

#P = rotmat(np.pi/2)
#Of = O.flatten()
#F = P@F
#Generating lines
#x_A = P@line_norm(n,c,k1,k2)+ Of[:,np.newaxis]#directrix
#x_B = P@line_norm(n,cl[0],k1,k2)+ Of[:,np.newaxis]#latus rectum
#print(n,c)
#xStandard =np.block([[x],[y]])

#Affine conic generation
#xActual = P@xStandard + Of[:,np.newaxis]

#x=np.arange(0,3.1,0.1) #range of values to shade the region
x=np.linspace(A[0],B[0],num).flatten() #range of values to shade the region
#shading the region
a = ab[0]
b = ab[1]
y1=(b/a)*(np.sqrt(a**2-x**2))
y2=(b/a)*(a-x)
plt.fill_between(x,y1,y2,color='green', alpha=.2)
#plotting
#plt.plot(xActual[0,:],xActual[1,:],label='Parabola',color='r')
plt.plot(xStandard[0,:],xStandard[1,:],label='Ellipse',color='r')
#plt.plot(xActual[0,:],xActual[1,:],label='Ellipse')
#plt.plot(x[0,:],x[1,:],label='Circle',color='g')
plt.plot(x_A[0,:],x_A[1,:],label='Chord')
#plt.plot(x_B[0,:],x_B[1,:],label='Chord')
#plt.plot(x,x,label='Chord')
#y1 =1 
#plt.fill_between(x,x,y,where= y < x, color = 'cyan', label = '$Area$')
#plt.fill_between(x,x1,y,where= (0< x)&(x < 1), color = 'cyan', label = '$Area$')
#plt.fill_between(xStandard[0,:],xStandard[1,:], where= (xStandard[0,:]>0 )&(xStandard[1,:]>x_A[1,:] ), color = 'cyan')
#plt.fill_between(x[0,:],x[1,:],where= (x[0,:] >=  0)&(x[1,:] >=  0)&(x[1,:] <=  A[1]), color = 'cyan', label = '$Area$')
#plt.plot(x_B[0,:],x_B[1,:],label='Latus Rectum')
#ax.fill_betweenx(y_fill, x_fill_parabola, x_fill_chord, where=(x_fill_chord < x_fill_parabola), color='cyan', alpha=0.5)

#
colors = np.arange(1,3)
#Labeling the coordinates
tri_coords = q
#tri_coords = np.block([A,B])
plt.scatter(tri_coords[0,:], tri_coords[1,:], c=colors)
vert_labels = ['$\mathbf{A}$','$\mathbf{B}$']
#vert_labels = ['$\mathbf{D}$','$\mathbf{E}$']
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
plt.xlabel('$x$')
plt.ylabel('$y$')
'''
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('chapters/12/8/3/8/figs/fig-temp.pdf')
subprocess.run(shlex.split("termux-open chapters/12/8/3/8/figs/fig-temp.pdf"))
#else
#plt.show()
