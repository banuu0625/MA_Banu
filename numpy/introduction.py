import numpy as np  #Numpy - Numeric Python

#ndarray = n-dimensional array

list1 =[1,2,3,4,5]


list2 = ["John Elder", 42, True]
print(list2)

np1 = np.array([0,1,2,3,4,5,6,7,8,9])
print(np1)


print(np1.shape)

#Range
np2 = np.arange(10)
print(np2)

#Step

np3 = np.arange(0,10,2)
print(np3)

#Zeros

np4 = np.zeros(10)
print(np4)

#Multudimensional zeros
np5 = np.zeros((2,10))
print(np5)

#Convert py list in np

my_list = [1,2,3,4,5]
np8 = np.array(my_list)
print(np8)