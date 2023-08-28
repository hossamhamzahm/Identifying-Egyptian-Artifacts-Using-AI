#imports
import time
import torch
import torch.nn as nn #neural network module
import torch.optim as optim #optimizers (e.g. Gradient Descent Stochastic Gradient Descent (SGD))
import torch.nn.functional as F # contains activation functions
from torch.utils.data import DataLoader # to load data set
import torchvision.datasets as datasets # to download dataset form mnisc
import torchvision.transforms
from torchvision.datasets import ImageFolder


def timer(f, txt="time"):
    def wrapper(*args):
        tic = time.time()
        val = f(*args)
        tac = time.time() - tic

        print(txt, tac, "ms")
        return val
    return wrapper


# Example transformations
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((450, 450)),  # Resize the image to (450, 450)
    torchvision.transforms.ToTensor(),          # Convert the PIL image to a PyTorch tensor (0-1)
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])




class CNN(nn.Module):
    def __init__(self, input_size=16*111*111, in_channels=3, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(input_size, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # keeping same number of examples and flaten the rest of dimensions
        x = x.reshape(x.shape[0], -1) 
        x = self.fc1(x)

        return x



# model = CNN()
# x = torch.randn(64, 3, 450, 450)
# print(model(x).shape)


# set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#hyperparameters
input_size = 16*111*111 #607500 # n. of input features
num_classes = 5 # n. of output classes
learning_rate = 0.001
#depicts the number of samples that propagate through 
#the neural network before updating the model parameters.
#the nn parameters are updated after each batch processed
batch_size = 64*2*2
num_epoches = 1 # how many times the entire dataset is fed to the network
training_image_path = "/kaggle/input/monuments-v0/monuments/train"
testing_image_path = "/kaggle/input/monuments-v0/monuments/test"


#load data
# train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_dataset = ImageFolder(root=training_image_path, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(root=testing_image_path, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
print("loaded data")


from torchvision.transforms import AutoImageProcessor, ResNetForImageClassification
import torch


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(train_loader, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

""" 
#initialize network
model = CNN(input_size=input_size, num_classes=num_classes).to(device)
print("initialized network")


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("created loss and optimizer")


# training the network
@timer
def train():
    for epoch in range(num_classes):
        for batch_idx, (data, targets) in enumerate(train_loader):

            # get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            #perform forward prop
            scores = model(data).to(device)
            loss = criterion(scores, targets).to(device)


            # back prop
            optimizer.zero_grad() # i don't know why this line
            loss.backward()

            #gradient descent or Adam step
            optimizer.step()
            # print(data.shape)


train()
            

@timer
def chech_accuracy(loader, model):
    num_correct=0
    num_samples=0
    model.eval()


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)

            
            num_correct += (y == predictions).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)}')
    
    model.train()


chech_accuracy(train_loader, model)
chech_accuracy(test_loader, model)


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"" """