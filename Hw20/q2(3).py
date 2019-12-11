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

train.data = train.data[:10000]
train.targets = train.targets[:10000]
# train_p = []
# for label in train.targets:
#   p = np.zeros(10)
#   p[label] = 1
#   train_p.append(p)
# train.targets = torch.tensor(train_p)

# test_p = []
# for label in test.targets:
#   p = np.zeros(10)
#   p[label] = 1
#   test_p.append(p)
# test.targets = torch.tensor(test_p)

batch_size = 100

train_batch = torch.utils.data.DataLoader(dataset = train,
                                             batch_size = batch_size,
                         
                                             shuffle = True)

test_batch = torch.utils.data.DataLoader(dataset = test,
                                            batch_size = 60000, 
                                      shuffle = False)


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(28,28)


train_data = images_to_vectors(train.data).float()[:100]
train_labels = train.targets[:100]


class Encoder(nn.Module):
  def __init__(self,encoding_length):
    super(Encoder,self).__init__()
    input_size = 784
    output_size = encoding_length

    self.layer1 = nn.Sequential(
      nn.Linear(input_size,1000),
      nn.ReLU())

    self.layer2 = nn.Sequential(
      nn.Linear(1000,output_size),
      nn.ReLU())

  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x



class Decoder(nn.Module):
  def __init__(self,encoding_length):
    super(Decoder,self).__init__()
    input_size = encoding_length
    output_size = 784

    self.layer1 = nn.Sequential(
      nn.Linear(input_size,1000),
      nn.ReLU())

    self.layer2 = nn.Sequential(
      nn.Linear(1000,output_size),
      nn.ReLU())

  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x

criterion = nn.MSELoss()
encoder = Encoder(50)
decoder = Decoder(50)
optimizer_encoder = torch.optim.Adam(encoder.parameters(),lr = 0.0001)
optimizer_decoder = torch.optim.Adam(decoder.parameters(),lr = 0.0001)
losses = []
epochs = 1000
def train_(data):
  optimizer_encoder.zero_grad()
  optimizer_decoder.zero_grad()
  encoding = encoder(data)
  decoding = decoder(encoding)
  error = criterion(decoding,data)
  losses.append(error.item())
  error.backward()
  optimizer_encoder.step()
  optimizer_decoder.step()


x = train_data[1]
plt.imshow(vectors_to_images(x),cmap = "gray")
plt.pause(2)
for epoch in range(epochs):
  print("epoch =",epoch)
  for n_batch,(real_batch,labels) in enumerate(train_batch):
    data = images_to_vectors(real_batch)
    train_(data)
    if epoch % 10 == 0:
      plt.cla()
      reconstructed = decoder(encoder(x))
      plt.imshow(vectors_to_images(reconstructed).detach(),cmap = "gray")
      plt.pause(0.001)
    break

reconstructed = decoder(encoder(x))
# print(vectors_to_images(x).shape)
plt.imshow(vectors_to_images(reconstructed).detach(),cmap = "gray")
plt.show()
plt.plot(losses)
plt.show()