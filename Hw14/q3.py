import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)
def main():
    train_data, train_labels = read_data("sample_train.csv")
    test_data, test_labels = read_data("sample_test.csv")

    class1 = []
    class2 = []

    for i in range(len(train_labels)):
        if(train_labels[i]==1):
            class1.append(train_data[i])
        elif(train_labels[i]==2):
            class2.append(train_data[i])
    test1 = []
    test2 = []
    for i in range(len(test_labels)):
        if(test_labels[i]==1):
            test1.append(test_data[i])
        elif(test_labels[i]==2):
            test2.append(test_data[i])
    class1 = np.array(class1)
    class2 = np.array(class2)
    test1 = np.array(test1)
    test2 = np.array(test2)

    labels1 = [ i for i in train_labels if i==1]
    labels2 = [ i for i in train_labels if i==2]
    testlabels1 = [ i for i in test_labels if i==1]
    testlabels2 = [ i for i in test_labels if i==2]

    trainclass = np.vstack((class1,class2))
    testclass = np.vstack((test1,test2))
    trainlabels = np.hstack((labels1,labels2))
    testlabels = np.hstack((testlabels1,testlabels2))

    Model = SVC(kernel='linear',C=1)
    Model.fit(trainclass,trainlabels)

    ct = 0
    # preds = []
    for i in range(len(testclass)):
        cur_pred = Model.predict(testclass[i].reshape(1,-1))
        # preds.append(cur_pred)
        if(cur_pred==testlabels[i]):
            ct+=1
    print("Accuracy is == ",ct/2)


    ###Part2 
    prince  = PCA(n_components=2)
    prince.fit(trainclass)
    transformed = prince.transform(trainclass)
    comps = prince.components_
    for i in range(len(transformed)):
        if(trainlabels[i]==1):
            plt.scatter(transformed[i][0],transformed[i][1],c='r')
        elif(trainlabels[i]==2):
            plt.scatter(transformed[i][0],transformed[i][1],c='b')
    w = Model.coef_
    w = np.array(w)
    w = np.dot(comps,w.T)
    x = np.linspace(-500,500,2)
    c = Model.intercept_
    y = -(w[0]*x+c)/(w[1])
    plt.plot(x,y,c='green')
    plt.show()

    ##part3
    vects = Model.support_vectors_
    prince = PCA(n_components=2)
    prince.fit(vects)
    vects2d = prince.transform(vects)
    comps_1 = prince.components_
    for i in vects2d:
        plt.scatter(i[0],i[1],c='y')
    plt.show()


main()