# ERA-Session5
ERA course Session 5 Assignment - Introduction to PyTorch

## Assignment Problem Description
Re-look at the code that we worked on in Assignment 4 (the fixed version). 
Move the contents of the code to the following files:
* model.py
* utils.py
* S5.ipynb

Make the whole code run again. 

Upload the code with the 3 files + README.md file (total 4 files) to GitHub. README.md (look at the spelling) must have details about this code and how to read your code (what file does what). Heavy negative scores for not formatting your markdown file into p, H1, H2, list, etc.

### Overall code details
Code consists of 11 Blocks - their use and to which file these codes goes are mentioned below.

#### 1. Code Block 1 - Importing necessary libraries and dependencies --> goes to `S5.ipynb`
```angular2html
################## CODE BLOCK 1 ###############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

########### Importing modules #################################################
import model as Model
import utils as Utils
```
#### 2. Code Block 2 - Checks if GPU is available or not, if not available CPU is used --> goes to `S5.ipynb`
```angular2html
################# CODE BLOCK 2 ################################################
# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
```
#### 3. Code Block 3 - Transforms Train and Test Datasets --> goes to `utils.py`
```angular2html
######### CODE BLOCK 3 ###############################################################
# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
```
#### 4. Code Block 4 - Downloads MNIST data if not present in the local and transforms the train and test data --> goes to `S5.ipynb`
```angular2html
################ CODE BLOCK 4 #################################################
train_data = datasets.MNIST('../data', train=True, download=True, transform=Utils.train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=Utils.test_transforms)
```
#### 5. Code Block 5 - Defines Batch size,Shuffle the data or not, number of cores to be used, memory and defines Train and Test Loaders --> goes to `S5.ipynb`
```angular2html
############### CODE BLOCK 5 ##################################################
batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
```
#### 6. Code Block 6 - Loads first batch of train data and plots first 12 MNIST images --> DATA CHECK/VISUALIZATION --> goes to `S5.ipynb`
``` angular2html
########### CODE BLOCK 6 ######################################################
import matplotlib.pyplot as plt

batch_data, batch_label = next(iter(train_loader)) 

fig = plt.figure()

for i in range(12):
  plt.subplot(3,4,i+1)
  plt.tight_layout()
  plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  plt.title(batch_label[i].item())
  plt.xticks([])
  plt.yticks([])
```
#### 7. Code Block 7 - Defines Neural Network Model Class --> goes to `model.py`
```angular2html
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
#### 7. Part of Code Block 7 - Model Summary showing the layers and the number of parameters each layer and total number of parameters --> goes to `S5.ipynb`
```angular2html
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Model.Net().to(device)
summary(model, input_size=(1, 28, 28))
```
#### 8. Code Block 8 - Defining Lists that capture Accuracy and Losses of train and test data --> goes to `S5.ipynb`
```angular2html
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
```
#### 9. Code Block 9 - train and test models with slight modifications where we return accuracy and loss values --> goes to `utils.py`
```angular2html
############### CODE BLOCK 9 #################################################
from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion): #,train_losses,train_acc):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

#  train_acc.append(100*correct/processed)
#  train_losses.append(train_loss/len(train_loader))
  train_acc = 100*correct/processed
  train_losses = train_loss/len(train_loader)
  return train_losses, train_acc

def test(model, device, test_loader, criterion): #, test_losses,test_acc):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_losses = test_loss

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses, test_acc
 ```
 #### 10. Code Block 10 - Defines hyper parameters and runs the Nerual Network model for 20 epochs and collects losses and accuracy --> goes to `S5.ipynb`
 ```angular2html
 ############### CODE BLOCK 10 #################################################
model =Model.Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = F.nll_loss
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train_losses_, train_acc_ = Utils.train(model, device, train_loader, optimizer, criterion)#,train_losses=train_losses,train_acc=train_acc)
  test_losses_, test_acc_ = Utils.test(model, device, test_loader, criterion)#,test_losses=test_losses,test_acc=test_acc)
  scheduler.step()

  train_losses.append(train_losses_)
  train_acc.append(train_acc_)
  test_losses.append(test_losses_)
  test_acc.append(test_acc_)
  ```
  #### 11. Code Block 11 - Plots training and Test sets loss and accuracy variation with epochs --> goes to `S5.ipynb`
  ```angular2html
  ############# CODE BLOCK 11 #################################################################
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")
```
![image](https://github.com/phaninandula/ERA-Session5/assets/30425824/726ff6d4-65e9-47bf-a168-aab5266863c1)
