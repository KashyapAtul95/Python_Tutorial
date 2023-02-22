# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:22:26 2023

@author: virtu
"""

import numpy as np

a = np.array([1,2,3])
print(a)

list1 = [1,2,3,4]
type(list1)

y = np.array(list1)
print(y)
type(y)
y
print(type(y))
print(y.ndim)


l = []

for i in range(1,5):
    int_1 = input("Enter :")       ##We get always strings in Input function
    l.append(int_1)
    
print(np.array(l))

ar2 = np.array([[1,2,3,4], [1,2,3,4]])
print(ar2)
print(ar2.ndim)

ar3 = np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
print(ar3)
print(ar3.ndim)


#n-dim array

arn = np.array([1,2,3,4], ndmin = 10)
arn
print(arn.ndim)

#Zeros Array

import numpy as np

ar_zero = np.zeros(4)
print(ar_zero)

ar_zero1 = np.zeros((3,4))
print(ar_zero1)

#ones Array

ar_one = np.ones(4)
print(ar_one)

ar_one1 = np.ones((4,4))
print(ar_one1)

#Empty
ar_em = np.empty(4)
print(ar_em)     #Empty prints last stored data in array.

#Range

ar_rn = np.arange(4)
print(ar_rn)

#Array diagonal elements filles with 1's::Identity matrix

ar_dig = np.eye(3)
print(ar_dig)

ar_dig1 = np.eye(3,5)
print(ar_dig1)

#Linspace

ar_lin = np.linspace(0,20,num = 5)
print(ar_lin)

#Numpy Arrays with random numbers

#rand():: 0 to 1

var = np.random.rand(4)
print(var)

var1 = np.random.rand(2,5)
print(var1)

#randn():: random numners close to zer, positive and negative

var2 = np.random.randn(5)
print(var2)

#ranf()

var3 = np.random.ranf(4)
print(var3)

#randint(min, max, number of samples)

var4 = np.random.randint(5,20,5)
print(var4)

#Data type in numpy

var_d = np.array([1,2,3,4])
print("Data Type: ",var_d.dtype)

var_d1 = np.array([1.0,2.1,3.2,4.3])
print("Data Type: ",var_d1.dtype)

var_d2 = np.array(["a", "b", "c", "d"])
print("Data Type: ",var_d2.dtype)

#Arithmetic operations in numpy array

import numpy as np

v1 = np.array([1,2,3,4])
varadd = v1+3
print(varadd)

v2 = np.array([1,2,3,4])
v3 = np.array([1,2,3,4])
varadd1 = v2+v3
print(varadd1)

varadd1_1 = np.add(v2,v3)
print(varadd1_1)

#2D Array

var21 = np.array([[1,2,3,4], [1,2,3,4]]) 
var22 = np.array([[1,2,3,4], [1,2,3,4]]) 

varadd2 = var21+var22
print(varadd2)
varprod1 = var21*var22
print(varprod1)

#Reciprocal

var_reci = np.array([1,2,3,4])
varadd_reci = np.reciprocal(var_reci)

#Arithmetic Functions

import numpy as np
var = np.array([1,2,3,4,5,3,2])
print("min: ", np.min(var))
print("max: ", np.max(var))


print("min: ", np.min(var), np.argmin(var))  #Index of min
print("max: ", np.max(var),np.argmax(var))   #Index of max

var1 = np.array([[2,1,3], [9,5,6]])
print(var1)

#Axis::0==Col
#Axis::1==Row

print(np.min(var1, axis=1))
print(np.min(var1, axis=0))
print("sqrt: ", np.sqrt(var1))

print("Sin: ", np.sin(var1))
print("Cos: ", np.sin(var1))

#cumulative Sum
var3 = np.cumsum(var)   
print(var3)

var4 = np.cumsum(var1)
print(var4)
type(var4)

#Shape and Reshape

import numpy as np

var = np.array([[1,2],[1,2]])
print(var)
print()
print(var.shape)

var1 = np.array([1,2,3,4], ndmin = 4)
print(var1)
print(var1.ndim)
print(var1.shape)

var2 = np.array([1,2,3,4,5,6])
print(var2.ndim)
x = var2.reshape(3,2)
print(x)
print(x.ndim)

#broadcasting

import numpy as np

var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3])

print(var1 + var2)

#Indexing and Slicing

#Indexing
import numpy as np

var = np.array([9,8,7,6])
print(var[0])
print(var[-1])

var1 = np.array([[9,8,7],[4,5,6]])
print(var1)
print(var1.ndim)

print(var1[0,0])

var2 = np.array([[[1,2],[6,7]]])
print(var2)
print(var2.ndim)
print()
print(var2[0,1,1])

#Slicing::start:Stop:Step

import numpy as np

var = np.array([1,2,3,4,5,6,7])
print(var)

print("2 to 5 : ",var[1:5])
print("2 to End : ",var[1:])
print("Start to 5 : ",var[:5])

print("Stop :", var[::2])

var1 = np.array([[1,2,3,4,5], [9,8,7,6,5], [11,12,13,14,15]])
print(var1)
print()
print("8 to 5 : ", var1[1,0:])
print("11 to 15 : ", var1[2,1:])

#Iteration

import numpy as np

var = np.array([9,8,7,6,5,4])
print(var)
print()

for i in var : 
    print(i)

var1 = np.array([[1,2,3,4],[1,2,3,4]])
print(var1)
print()

for j in var1:
    print(j)

print()

for k in var1:
    for l in k:
        print(l)


var3 = np.array([[[9,8,7,6],[1,2,3,4]]])
print(var3)
print(np.ndim(var3))

for i in var3:
    for j in i:
        for l in j:
            print(l)


var4 = np.array([[[9,8,7,6],[1,2,3,4]]])
print(var4)
print(np.ndim(var4))
print()

for i in np.nditer(var4):
    print(i)
    
# Iteration with indexing:

var5 = np.array([[[9,8,7,6],[1,2,3,4]]])
print(var5)
print(np.ndim(var5))
print()

for i,d in np.ndenumerate(var4):
    print(i,d)

#Copy vs View

import numpy as np

#Copy:: Copy owns data, any changes made to the copy will not affect the original array 

var = np.array([1,2,3,4])
co = var.copy()
print("Var : ", var)
print("Copy : ", co)


#View:: View does not own data, any changes made to the view will affect the original array 

x = np.array([9,8,7,6,5])
vi = x.view()

print("x : ", x)
print("view : ", vi)

#join Array

import numpy as np

var = np.array([1,2,3,4])
var1 = np.array([9,8,7,6])

ar = np.concatenate((var, var1))
print(ar)

vr = np.array([[1,2],[3,4]])
vr1 = np.array([[9,8],[7,6]])

ar_new = np.concatenate((vr, vr1), axis = 1)
print(ar_new)
print()

ar_new1 = np.concatenate((vr, vr1), axis = 0)
print(ar_new1)

var_1 = np.array([1,2,3,4])
var_2 = np.array([9,8,7,6])

a_new = np.stack((var_1, var_2), axis = 1)
print(a_new)

a_new1 = np.hstack((var_1, var_2))     #Along rows
print(a_new1)
print()
a_new2 = np.vstack((var_1, var_2))     #Along Columns
print(a_new2)
print()
a_new3 = np.dstack((var_1, var_2))     #Along Heights
print(a_new3)
print()

#Split Array
import numpy as np

var = np.array([1,2,3,4,5,6])
print(var)

ar = np.array_split(var, 3)
print(ar)
print(type(ar))
print()

print(ar[0])
print(ar[1])
print(ar[2])

var1 = np.array([[1,2],[3,4],[5,6]])
print(var1)
ar1 = np.array_split(var1, 3, axis = 1)
print(ar1)
print()

ar2 = np.array_split(var1,3)
print(ar2)

#Numpy arrays functions

#Search

import numpy as np

var = np.array([1,2,3,4,2,5,2,5,6,7])

x = np.where(var == 2)
print(x)
print()

x1 = np.where(var/2 == 2)
print(x1)

#Search Sorted Array

var1 = np.array([1,2,3,4,6,7,8])

x2 = np.searchsorted(var1, 5)
print(x2)
print()

x2_1 = np.searchsorted(var1, 5, side="right")
print(x2_1)
print()

#Sort

var4 = np.array([1,2,3,4,2,5,2,5,6,7])
print(np.sort(var4))

#Filter Array

var5 = np.array(["a", "s", "d", "f"])
print(var5)
print()
f = [True,False,False,True]
new_a = var5[f]
print(new_a)
print(type(new_a))

#Shuffle

vb = np.array([1,2,3,4,5])
np.random.shuffle(vb)
print(vb)

#Unique

vb1 = np.array([1,2,3,4,2,5,2,2,6,7])
x1 = np.unique(vb1, return_index=True, return_counts=True)
print(x1)

#Resize
vb2 = np.array([1,2,3,4,5,6])
y = np.resize(vb2, (3,2))
y

#Flatten & Ravel
vb2 = np.array([1,2,3,4,5,6])
y = np.resize(vb2, (3,2))
print(y.flatten())
print()

print("Flatten : ",y.flatten(order="F"))
print("Ravel : ",y.ravel(order="F"))

#Insert and Delete Arrays
import numpy as np

var = np.array([1,2,3,4])
print(var, var.ndim, type(var))

np.insert(var, 2, 400)
np.insert(var, (2,4), (999,999))
#Note: Insert function does not accept float values.

var1 = np.array([[1,2,3],[1,2,3]])
v1 = np.insert(var1, 2, 6, axis = 0)
print(v1)

#Delete
var_d = np.array([1,2,3,4])
print(var_d)

d=np.delete(var_d, 2)
print(d)

#Matrix in Numpy Arrays

import numpy as np

var = np.matrix([[1,2],[1,2]])
var2 = np.matrix([[1,2],[1,2]])
print(var, type(var))
print(var+var2)
print(var-var2)
print()
print(var.dot(var2))

var1 = np.array([[1,2,3],[1,2,3]])
var3 = np.array([[1,2,3],[1,2,3]])
print(var1, type(var1))
print(var1*var3)

#Transpose, Swapaxes, Inverse, Power, Determinate

import numpy as np
var = np.matrix([[1,2,3],[4,5,6]])
print(np.transpose(var))
print(var.T)
print()

print(np.swapaxes(var, 0, 1))

var2 = np.matrix([[1,2],[3,4]])
print(var2)
print()
print(np.swapaxes(var2, 0, 1))

var3 = np.matrix([[1,2],[3,4]])
print(np.linalg.inv(var3))
print()

print(var3.dot(var3))     #Matrix Multiplication by dot function


var4 = np.matrix([[1,2],[3,4]])
print(np.linalg.matrix_power(var4,2))
print()

print(np.linalg.matrix_power(var4,0))
print()

print(np.linalg.matrix_power(var4,-2))
print()

#Determinant

var5 = np.matrix([[1,2],[3,4]])
print(np.linalg.det(var5))





