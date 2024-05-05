import numpy as np

#slicing Numpy Arrays

np1 = np.array([1,2,3,4,5,6,7,8,9])

#return 2,3,4,5

print(np1[1:5])

#return from somethig till end

print(np1[3:])

#return negative slice

print(np1[-3:-1])

#steps

print(np1[1:5:2])

#steps on entire array

print(np1[::2])

#slice 2D-array

np2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

#pull out a single item

print(np2[1,2])

#SCLICE 2D array
print(np2[0:1, 1:3])

print(np2[0:2, 1:3])