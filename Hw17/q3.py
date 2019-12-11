import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy.linalg as LA

err = []
for i in range(1,20):
    alpha = np.random.uniform(0,1,i)
    alpha = alpha/(np.sum(alpha))
    sigma = 1
    points = 800
    x = np.random.uniform(15,25,i)
    for j in range(points):
        n = np.random.normal(0,sigma)
        one_ahead = x[::-1][:i].T @ alpha + n
        x = np.append(x,one_ahead)
    X = []
    Y = []
    for j in range(len(x)-i):
        Y.append(x[j+i])
        X.append(x[:j+i][::-1][:i])
    X = np.array(X)
    Y = np.array(Y)
    model = LinearRegression().fit(X,Y)
    plt.plot(Y)
    mypredictions = model.predict(X)
    cur_err = LA.norm((Y-mypredictions))
    err.append(cur_err)
    plt.plot(mypredictions)
    acc = model.score(X,Y)
    acc = acc*100
    plt.title("For d value == " + str(i) + "  The accuracy is " + str(acc))
    # plt.show()
plt.show()

plt.plot([i for i in range(1,20)],err)
plt.title("The error plot versus d values")
plt.show()



