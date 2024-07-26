#Code by GVV Sharma
#September 21, 2012
#Revised July 26, 2024
#Released under GNU/GPL

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

#Creating matrices
m1 = np.array(([1,-3,2])).reshape(-1,1)
m2 = np.array(([2,3,1])).reshape(-1,1)
P1 = np.array(([1,2,3])).reshape(-1,1)
P2 = np.array(([4,5,6])).reshape(-1,1)
b = P2-P1
A = np.block([m1,m2])
ans = 1/19*np.array(([10,28])).reshape(-1,1)

#SVD, A=USV
U, s, V=LA.svd(A)

#Find size of A
mn=A.shape

#Creating the singular matrix
S = np.zeros(mn)
Sinv = S.T
S[:2,:2] = np.diag(s)

#Verifying the SVD, A=USV
#print(U@S@V)
print(U,s,V)

#Inverting s
sinv = 1./s
#Inverse transpose of S
Sinv[:2,:2] = np.diag(sinv)
#print(Sinv)

#Moore-Penrose Pseudoinverse
Aplus = V.T@Sinv@(U.T)

#Least squares solution
x_ls = Aplus@b
#
print(x_ls,ans)
