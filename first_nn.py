#imports
import torch
import torch.nn as nn #neural network module
import torch.optim as optim #optimizers (e.g. Gradient Descent Stochastic Gradient Descent (SGD))
import torch.nn.functional as F # contains activation functions
from torch.utils.data import DataLoader # to load data set
import torchvision.datasets as datasets # to download dataset form mnisc
import torchvision.transforms as transforms 


#create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50) # input la yer with output = 50
        self.fc2 = nn.Linear(50, num_classes) # hidden layer with input = 50 and output = n. of classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#set device 
device = "cpu" #torch.device("cuda" if torch.cuda_is_available() else "cpu")


#hyperparameters
input_size = 784 # n. of input features
num_classes = 10 # n. of output classes
learning_rate = 0.001
#depicts the number of samples that propagate through 
#the neural network before updating the model parameters.
#the nn parameters are updated after each batch processed
batch_size = 64 
num_epoches = 1 # how many times the entire dataset is fed to the network


#load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, )
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, )



#initialize network
model = NN(input_size=input_size, num_classes=num_classes)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training the network
for epoch in range(num_classes):
    for batch_idx, (data, targets) in enumerate(train_loader):

        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # flaten the features
        data = data.reshape(data.shape[0], -1)
        
        #perform forward prop
        scores = model(data)
        loss = criterion(scores, targets)


        # back prop
        optimizer.zero_grad() # i don't know why this line
        loss.backward()
        
        #gradient descent or Adam step
        optimizer.step()
        # print(data.shape)




def chech_accuracy(loader, model):
    num_correct=0
    num_samples=0
    model.eval()


    with torch.no_grad():
        for x, y in loader:
            x.to(device=device)
            y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)

            
            num_correct += (y == predictions).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)}')
    
    model.train()


chech_accuracy(train_loader, model)
chech_accuracy(test_loader, model)