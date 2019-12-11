import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def images_to_vectors(images):
	return images.view(images.size(0),784)



train_dataset = datasets.MNIST(root = './data',train = True,transform = transforms.ToTensor())
train_dataset.data = train_dataset.data[:10000]
train_dataset.targets = train_dataset.targets[:10000]


batch_size = 100

train_batch = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)

train_data = images_to_vectors(train_dataset.data).float()[:100]
train_labels = train_dataset.targets[:100]

class Net(nn.Module):
	def __init__(self,weight_type = None):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(nn.Linear(784,1000),nn.ReLU())
		self.layer2 = nn.Sequential(nn.Linear(1000,1000),nn.ReLU())
		self.layer3 = nn.Sequential(nn.Linear(1000,10),nn.Sigmoid())

		def init_weights(m):
			if type(m) == "Linear":
				if weight_type == "uniform":
					torch.nn.init.uniform_(m.weight,0,1)
				elif weight_type == "normal":
					torch.nn.init.normal_(m.weight,0,1)
				elif weight_type == "xavier":
					torch.nn.init.xavier_uniform_(m.weight)


		if weight_type != None:
			self.layer1.apply(init_weights)
			self.layer2.apply(init_weights)
			self.layer3.apply(init_weights)


	def forward(self,x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		return x



criterion = nn.CrossEntropyLoss()
nets = [Net(),Net(),Net(),Net(),Net(),Net()]
optimizers = [torch.optim.SGD(nets[0].parameters(),lr = 10),torch.optim.SGD(nets[1].parameters(),lr = 1),torch.optim.SGD(nets[2].parameters(),lr = 0.1),
			torch.optim.SGD(nets[3].parameters(),lr = 0.01),torch.optim.SGD(nets[4].parameters(),lr = 0.0001),torch.optim.SGD(nets[5].parameters(),lr = 0.00001)]
losses = [[],[],[],[],[],[]]

def train_(data,labels):
	for i in range(len(nets)):
		losses[i].append(criterion(nets[i](train_data),train_labels).item())
		optimizers[i].zero_grad()
		output = nets[i](data)
		error = criterion(output, labels)
		error.backward()
		optimizers[i].step()


epochs = 200

for epoch in range(epochs):
	if epoch%25 == 0:
		print("epoch = {}/{}".format(epoch,epochs))

	for n_batch,(real_batch,labels) in enumerate(train_batch):
		data = images_to_vectors(real_batch)
		train_(data,labels)
		break


learning_rates = [10,1,0.1,0.01,0.0001,0.00001]
for i in range(len(nets)):
	plt.plot(losses[i],label = str(learning_rates[i]))
plt.title("Loss function with varing learning rates")
plt.legend()
plt.show()


criterion = nn.CrossEntropyLoss()
nets = [Net(weight_type = "uniform"),Net(weight_type = "normal"),Net(weight_type = "xavier")]
optimizers = [torch.optim.SGD(nets[0].parameters(),lr = 1),torch.optim.SGD(nets[1].parameters(),lr = 1),torch.optim.SGD(nets[2].parameters(),lr = 1)]
losses = [[],[],[]]


epochs = 200

for epoch in range(epochs):
	if epoch%25 == 0:
		print("epoch = {}/{}".format(epoch,epochs))

	for n_batch,(real_batch,labels) in enumerate(train_batch):
		data = images_to_vectors(real_batch)
		train_(data,labels)
		break

weight_types = ["uniform","normal","xavier"]
for i in range(len(nets)):
	plt.plot(losses[i],label = weight_types[i])
plt.title("Loss function with varing initial_weights")
plt.legend()
plt.show()
