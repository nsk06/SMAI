import numpy as np
import matplotlib.pyplot as plt

def samplevariance(s,k):
    # S = [np.random.normal(0,1,k)]*s
    S = []
    for i  in range(s):
        S.append(np.random.normal(0,1,k))
    means = []
    S = np.array(S)
    for dist in S:
        means.append(np.mean(dist))
    return np.cov(means)

iter = 500
vars = []
s = 1000
for k in range(1,iter):
    vars.append(samplevariance(s,k))
plt.plot(vars)
plt.grid(True)
plt.title("Varying K")
plt.show()
k = 1000
vars = []
for s in range(1,iter):
    vars.append(samplevariance(s,k))
plt.plot(vars)
plt.grid(True)
plt.title("Varying S")
plt.show()