import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
N = 1000
X = np.concatenate([np.array([[i,i] for i in np.linspace(-100,100,N//2)]),np.array([[i,-i] for i in np.linspace(-100,100,N//2)])])
y = np.array([1 if i < N//2 else -1 for i in range(N)])
print(X.shape)
print(y.shape)
plt.scatter(X[:,0],X[:,1],c = y)
# Algorithm
x = [i for i in range(N)]
x = np.array(x)
np.random.shuffle(x)
y1_ = x[:N//2]
y2_ = x[N//2:N]

y1 = [i for i in y1_]
y2 = [i for i in y2_]

converged = False
iters = 0
max_iters = 100
while not converged:
  converged = True
  plt.title("Iter = "+str(iters))
  plt.scatter(X[:,0],X[:,1],c = [1 if i in y1 else 2 for i in range(N)])
  plt.pause(0.1)
  iters += 1
  X1 = X[y1_]
  X2 = X[y2_]

  mean1 = np.mean(X1,axis = 0)
  mean2 = np.mean(X2,axis = 0)

  d1,u1 = np.linalg.eig(np.cov(X1.T))
  u1 = u1[:,0]
  d2,u2 = np.linalg.eig(np.cov(X2.T))
  u2 = u2[:,0]

  for i in range(N):
    x = X[i,:]
    a1 = np.dot(u1,x - mean1)**2
    a2 = np.dot(u2,x - mean2)**2
    b1 = np.linalg.norm(x-mean1)**2 - a1
    b2 = np.linalg.norm(x-mean2)**2 - a2

    if b1 <= b2:
      if i not in y1:
        y1.append(i)
        y2.remove(i)
        converged = False
        
    else:
      if i not in y2:
        y2.append(i)
        y1.remove(i)
        converged = False
  y1_ = np.array(y1)
  y2_ = np.array(y2)
  if iters > max_iters:
    converged = True

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax2.scatter(X[:,0],X[:,1],c = y)
ax2.set_title("Original Data")

ax1.scatter(X[:,0],X[:,1],c = [1 if i in y1 else 2 for i in range(N)])
ax1.set_title("Converged output")
plt.show()