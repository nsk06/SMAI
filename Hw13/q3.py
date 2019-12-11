import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")

classwise = []
class_labels = []
for i in range(10):
    temp =[]
    for j in range(len(train_labels)):
        cur = train_labels[j]
        if(cur==i):
            temp.append(train_data[j])
    classwise.append(temp)
    class_labels.append([i]*len(temp))
classwise = np.array(classwise)
class_labels = np.array(class_labels)

paircls = []
for i in range(10):
    temp = []
    for j in range(10):
        if(i!=j):
            X = np.vstack((classwise[i],classwise[j]))
            Y = np.hstack((class_labels[i],class_labels[j]))
            temp.append(LogisticRegression(solver='liblinear').fit(X,Y))
        else:
            temp.append(-1)
    paircls.append(temp)
ct = 0
preds =[]
for i in range(len(test_data)):
    cur = test_data[i]
    carr = [0]*10
    for j in range(10):
        for k in range(j+1,10):
            carr[int(paircls[j][k].predict(cur.reshape(1,-1)))]+=1
    pred_l = carr.index(max(carr))
    preds.append(pred_l)
    if(pred_l==test_labels[i]):
        ct+=1
cm = confusion_matrix(test_labels,preds)
print(cm)
print("Accuracy is = ",ct/10)
cmp=plt.matshow(cm)
plt.colorbar(cmp)
plt.show()