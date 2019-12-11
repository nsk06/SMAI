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

clf = LogisticRegression(solver = 'lbfgs',multi_class = 'auto',max_iter = 1000).fit(train_data,train_labels)
ct = 0
preds = np.zeros(len(test_data))
for i in range(len(test_data)):
    cur_test = test_data[i]
    labels = test_labels[i]
    py = clf.predict(cur_test.reshape(1,-1))
    preds[i] = py
    if py == labels:
        ct+=1
cm = confusion_matrix(test_labels,preds)
print(cm)
cmp = plt.matshow(cm)
plt.colorbar(cmp)
plt.show()
print("Accuracy is = ",ct/10)
