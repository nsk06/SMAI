import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import copy

def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    for i in dict:
        if(i.decode('UTF-8')=='labels'):
            labels = dict[i]
        elif(i.decode('UTF-8')=='data'):
            data = dict[i]
    return data,labels
def classifydata(data,labels,ind):
    temp = []
    for i in range(len(labels)):
        if(labels[i]==ind):
            temp.append(data[:,i])
    temp = np.array(temp)
    temp = temp.T
    return temp
def topvectsclasses(classes,means):
    vects = []
    for i in range(10):
        avg_data = classes[i]-means[i]
        covmat = np.cov(avg_data)
        eigvals,eigvecs = LA.eig(covmat)
        inds = eigvals.argsort()[::-1]
        eigvals = eigvals[inds]
        eigvecs = eigvecs[:,inds]
        prine =  eigvecs[:,0:20]
        vects.append(prine)
    vects = np.array(vects)
    # print(vects.shape)
    return vects
def parta(classes,means,topvecs):
    errors = []
    for i in range(10):
        avg_data = classes[i]-means[i]
        prine =  topvecs[i]
        project = np.matmul(avg_data.T,prine)
        reconstruct = np.matmul(project,prine.T)
        reconstruct = reconstruct.T +means[i]
        errors.append(np.sqrt(np.mean((classes[i]-reconstruct)**2)))
    print(errors)
    plt.plot([i for i in range(10)],errors)
    plt.show()
def partb(means):
    mean_distances = []
    for i in range(10):
        dists = []
        for j in range(10):
            dists.append(LA.norm(means[i]-means[j]))
        mean_distances.append(dists)
        print(i,dists)
    # print(mean_distances)
def partc(classes,means,topvecs):
    Errors = []
    for i in range(10):
        my_errors = []
        for j in range(10):
            if i!=j:
                avg_data = classes[i]-means[i]
                prine =  topvecs[j]
                project = np.matmul(avg_data.T,prine)
                reconstruct = np.matmul(project,prine.T)
                reconstruct = reconstruct.T +means[i]
                my_errors.append(np.sqrt(np.mean((classes[i]-reconstruct)**2)))
            else:
                my_errors.append(-1) #for similarity
        # my_errors = sorted(my_errors,key=lambda x: x[0])
        Errors.append(my_errors)
    # print(Errors)
    ans_list = []
    for i in range(10):
        temp = []
        for j in range(10):
            if i!=j:
                temp.append(((Errors[i][j]+Errors[j][i])/2,j))
        temp = sorted(temp,key=lambda x: x[0])
        ans_list.append([temp[0][1],temp[1][1],temp[2][1]])
        print("Class :",i)
        print("First neighbour, ",temp[0][1])
        print("Second neighbour, ",temp[1][1])
        print("Third neighbour, ",temp[2][1])
        print("***************************************************************************************************************")
    # print(ans_list)

data_1,labels_1= unpickle('./cifar-10-batches-py/data_batch_1')
data_2,labels_2= unpickle('./cifar-10-batches-py/data_batch_2')
data_3,labels_3= unpickle('./cifar-10-batches-py/data_batch_3')
data_4,labels_4= unpickle('./cifar-10-batches-py/data_batch_4')
data_5,labels_5= unpickle('./cifar-10-batches-py/data_batch_5')
data = np.hstack((data_1.T,data_2.T,data_3.T,data_4.T,data_5.T))
# print(data[:,1].shape)
labels = np.hstack((np.array(labels_1).T,np.array(labels_2).T,np.array(labels_3).T,np.array(labels_4).T,np.array(labels_5).T))
classes = []
for i in range(10):
    classes.append(classifydata(data,labels,i))
classes = np.array(classes)
means = []
for i in classes:
    means.append(np.mean(i,axis=1).reshape(3072,1))
means = np.array(means)
# print(classes[0][:,0].reshape((32,32,3)).shape)
# plt.imshow(data_1[0,:].reshape((32,32,3)))
# plt.show()
# topvecs = topvectsclasses(classes,means)
# parta(classes,means,topvecs)
partb(means)
# partc(classes,means,topvecs)