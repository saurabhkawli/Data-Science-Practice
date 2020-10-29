import numpy as np
a=np.array([1,2,3,4,5])
print (a)
print (type(a))
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
print(a.itemsize)

#%%
b=np.array([(1,2,3,4,5),(6,7,8,9,0)])
print(b.shape)
print(b.size)
print(b.dtype)
print(b.itemsize)
b.ndim
type(a)

#%%

c=np.array([(0,1,2,3,4),(5,6,7,8,9),(10,11,12,13,14)])
c


#%%
#How to create multidimensional array

d=np.arange(120).reshape(4,5,6)
d
print(d.shape)
print(d.size)
print(d.dtype)
print(d.itemsize)
d.ndim
type(d)
#%%
e=np.ones((5,5))
e
f=np.zeros((5,5))
f
#%%
g=np.random.random((5,5))
g

#%%
h=np.full((2,2),7.5)
h
#%%
i=np.identity(10,dtype=np.int)
i
