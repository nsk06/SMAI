import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

train = datasets.MNIST(root = './data', train = True,
                        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ]), download = True)

test = datasets.MNIST(root = './data', train = False,
                       transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ]))

idx1 = train.targets == 1
idx2 = train.targets == 2
idx = idx1 + idx2
train.targets = train.targets[idx] 
train.data = train.data[idx]
train.targets = train.targets - 1

idx1 = test.targets == 1
idx2 = test.targets == 2
idx = idx1 + idx2
test.targets = test.targets[idx] 
test.data = test.data[idx]
# test.targets = test.targets - 1


# print(train.targets,train.targets - 1)
batch_size = 1
train_batch = torch.utils.data.DataLoader(dataset = train,
                                             batch_size = batch_size,
                         
                                             shuffle = True)

test_batch = torch.utils.data.DataLoader(dataset = test,
                                            batch_size = 60000, 
                                      shuffle = False)

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    input_size = 784
    output_size = 1
    self.layer1 = nn.Sequential(
        nn.Linear(input_size,1000),
        nn.ReLU()
      )
    self.layer2 = nn.Sequential(
        nn.Linear(1000,1000),
        nn.ReLU()
      )

    self.layer3 = nn.Sequential(
        nn.Linear(1000,output_size),
        # nn.functional.sigmoid()
        nn.Sigmoid()
      )

  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x

  def classify(self,x):
    x = self.forward(x.float())
    x *= 2
    x = x.int()
    x /= 2
    return x + 1
criterion = nn.BCELoss()
net = Net()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.0001)


def train_(data,labels):
  optimizer.zero_grad()
  output = net(data)

  error = criterion(output,labels.float())
  error.backward()
  optimizer.step()

epochs = 1000
# print(train.data.size())
for epoch in range(0,epochs):
  print("epoch =", epoch)

  for n_batch,(real_batch,labels) in enumerate(train_batch):
    # data = torch.autograd.Variable(images_to_vectors(real_batch))
    data = images_to_vectors(real_batch)
    # print(labels.size())
    # print(data.size())
    train_(data,labels)
    break



test_images = test.data
test_data = images_to_vectors(test_images)
test_labels = test.targets
plt.pause(0.01)
classification = net.classify(test_data).T
# plt.imshow(test_images[0],cmap = "gray")
# plt.show()
correct = (classification == test_labels)
print("accuracy = ", correct.float().mean().item())
