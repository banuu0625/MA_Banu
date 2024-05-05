import numpy as np

np1 = np.array([0,1,2,3,4,5,6,7,8,9])

#create a view

np2 = np1.view()

print(f'Original NP1 {np1}')
print(f'Original NP2 {np2}')

np1[0] = 41

print(f'Changed NP1 {np1}') #whenever change is made on orignal array, the changes will also be applied to the view
print(f'Original NP2 {np2}')

np1[0] = 0


np3 = np1.copy()

print(f'Original NP1 {np1}')
print(f'Original NP3 {np3}')

np1[0] = 41

print(f'Changed NP1 {np1}') #whenever change is made on orignal array, the changes will NOT be applied to the copy
print(f'Original NP3 {np3}')

