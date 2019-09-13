import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.model_selection import KFold
sz = 100
x = np.linspace(0,1,sz)
noise = np.random.normal(0,0.1,sz)
y = np.sin(x)+noise
means = []
deviations = []
klist = [i for i in range(2,sz-1)]
for k in range(2,sz-1):
    kf = KFold(n_splits=k)
    mean =[]
    stdv = []
    errors = []
    for train_set,test_set in kf.split(x):
        errors.append(np.sum(noise[test_set]**2)/len(test_set))
        mean.append(np.sum(noise[test_set]**2)/len(test_set))
        cur_mean = np.mean(noise[test_set])
        stdv.append(np.sqrt(np.sum((noise[test_set]-cur_mean)**2)/len(test_set)))
    mean = np.array(mean)
    stdv = np.array(stdv)
    errors = np.array(errors)
    means.append(np.mean(mean))
    deviations.append(np.mean(stdv))
means = np.array(means)
print(means.shape)
deviations = np.array(deviations)
klist = np.array(klist)
# print(klist)
plt.plot(klist,deviations,c='r')
plt.plot(klist,means,c='b')
# plt.plot(x,y)
plt.show()