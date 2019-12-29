import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from validation import validate
from input_args import get_input_args
from testing import test
from checkpoints_ops import save_chk

in_args=get_input_args('train')

data_dir = in_args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets =  datasets.ImageFolder(train_dir, transform=data_transforms)
test_datasets =  datasets.ImageFolder(test_dir, transform=data_transforms)
val_datasets =  datasets.ImageFolder(valid_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
val_loaders = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=True)
inputsize=0

if in_args.arch=='vgg':
    model=models.vgg16(pretrained=True)
    inputsize=25088
if in_args.arch=='alexnet':
    model= models.alexnet(pretrained=True)
    inputsize= 9216
    
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputsize, in_args.hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout()),
                          ('fc2', nn.Linear(in_args.hidden_units, in_args.hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout()),
                          ('fc3', nn.Linear(in_args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))           
                          ]))
    
model.classifier = classifier
criterion = nn.CrossEntropyLoss()
    # Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
print ("Creation done, starting training")
device=in_args.device
epochs=in_args.epochs
if device=='cuda':
    model.to(device)

steps = 0
for e in range(epochs):
        running_loss = 0
        for images, labels in iter(dataloaders):
            model.train()
            steps += 1
            print (images.size())
            #images.resize_(images.size()[0],inputsize)
            print(images.size())
            if device=='cuda':
                images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            validate(model,val_loaders,criterion,device)
            
            if steps % 32 == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/32))
                running_loss = 0
                
                
#for testing 
test(model,test_loaders,criterion,device)

#save checkpoints 
save_chk(model,epochs,in_args.save_dir)
