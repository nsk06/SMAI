import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

def main():
    points = 1000
    p1 = np.array([[i,i] for i in np.linspace(-100,100,500)])
    p2 = np.array([[i,-i] for i in np.linspace(-100,100,500)])
    # print(p1.shape)
    # print(p2.shape)
    data = np.vstack((p1,p2))
    # print(data.shape)
    y1 = [ 1 for i in range(500)]
    y2 = [-1 for i in range(500)]
    y = np.hstack((y1,y2))
    # print(y.shape)
    plt.subplot(211)
    plt.scatter(data[:,0],data[:,1],c=y)
    plt.title("Original Data")
    # plt.show()
    iters = 200
    inds = [i for i in range(points)]
    inds = np.array(inds)
    np.random.shuffle(inds)
    c1 = inds[:500]
    c2 = inds[500:points]
    r1 = [ i for i in c1]
    r2 = [i for i in c2]

    for i in range(iters):
        x1 = data[c1]
        x2 = data[c2]
        m1 = np.mean(x1,axis=0)
        m2 = np.mean(x2,axis=0)
        _,u1 = LA.eig(np.cov(x1.T))
        _,u2 = LA.eig(np.cov(x2.T))
        u1 = u1[:,0]
        u2 = u2[:,0]
        for j in range(points):
            x = data[j,:]
            k1 = np.dot(u1,x-m1)**2
            k2 = np.dot(u2,x-m2)**2
            b1 = LA.norm(x-m1)**2-k1
            b2 = LA.norm(x-m1)**2-k2
            if b1 <= b2:
                if j not in r1:
                    r1.append(j)
                    r2.remove(j)
            else:
                if j not in r2:
                    r2.append(j)
                    r1.remove(j)
        r1 = np.array(r1)
        r2 = np.array(r2)
    plt.subplot(212)
    plt.scatter(data[:,0],data[:,1],c = [1 if i in r1 else 2 for i in range(points)])
    plt.title("Output")
    plt.show() 


if __name__ == "__main__":
    main()
