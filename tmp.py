import numpy as np


e = np.arange(2)
a = np.arange(6).reshape(2,3)
b = np.arange(8).reshape(2,4)
c = np.einsum('bi,bj->ij', a, b) / 2
d = np.einsum('bi,bj->bij', a, b).mean(axis=0)

e = np.arange(2)
e = e[:, np.newaxis]
d = np.einsum('bi,bj->j', e, b)

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)

print(c)
print(d)

print('')
print(e)
print(b)
print(d)
