import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.optimizers import RMSprop,Adam
import keras.initializers
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
        if(train_labels[i]==0):
            class1.append(train_data[i])
        elif(train_labels[i]==1):
            class2.append(train_data[i])
    test1 = []
    test2 = []
    for i in range(len(test_labels)):
        if(test_labels[i]==0):
            test1.append(test_data[i])
        elif(test_labels[i]==1):
            test2.append(test_data[i])
    class1 = np.array(class1)
    class2 = np.array(class2)
    test1 = np.array(test1)
    test2 = np.array(test2)

    labels1 = [ i for i in train_labels if i==0]
    labels2 = [ i for i in train_labels if i==1]
    testlabels1 = [ i for i in test_labels if i==0]
    testlabels2 = [ i for i in test_labels if i==1]

    trainclass = np.vstack((class1,class2))
    testclass = np.vstack((test1,test2))
    trainlabels = np.hstack((labels1,labels2))
    testlabels = np.hstack((testlabels1,testlabels2))


    epochs = 18
    trainclass = trainclass.reshape(len(trainclass),28*28)
    testclass = testclass.reshape(len(testclass),28*28)
    testclass /= 255
    trainclass /= 255
    predtrain  = keras.utils.to_categorical(trainlabels,2)
    predtest  = keras.utils.to_categorical(testlabels,2)

    Model = Sequential()
    Model.add(Dense(1000, activation='tanh', input_shape=(784,),kernel_initializer=keras.initializers.Zeros()))
    Model.add(Dense(1000, activation='tanh',kernel_initializer=keras.initializers.Zeros()))
    Model.add(Dense(2, activation='sigmoid',kernel_initializer=keras.initializers.Zeros()))
    Model.summary()
    Model.compile(loss='MSE',
              optimizer=Adam(),
              metrics=['accuracy'])
    store = Model.fit(trainclass,predtrain,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(testclass, predtest))
    metric = Model.evaluate(testclass,predtest,verbose=0)
    print("Test Accuracy is : ",metric[1])


main()