import numpy as np
import matplotlib.pyplot as plt

m = 2

A = np.random.randn(m,m)
b = np.random.randn(m)
x = np.linalg.solve(A, b)

kappa = np.linalg.cond(A)

# print(f'{np.linalg.cond(A) = }')

p = 1000
delta = np.zeros((p,2))
for k in range(p):
	Ap = A + 1e-10 * np.random.randn(m,m)
	xp = np.linalg.solve(Ap, b)
	delta[k,:] = ((xp - x) / np.linalg.norm(x)) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1])
circle = plt.Circle((0.0,0.0), kappa, fill=False)
ax.add_patch(circle)

print(f'{kappa = }')
print(f'{np.max(np.linalg.norm(delta, axis=1)) = }')

plt.show()

