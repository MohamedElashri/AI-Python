import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from PIL import Image
import numpy as np

def process_image(imagedir):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(imagedir)
    print("loaded successfully")
    print(type(image))
    # Resize
    if image.size[0] > image.size[1]:
        image=image.resize((image.size[0]+2, 256))
    else:
        image=image.resize((256, image.size[1]+2))
        
    
    # Crop 
    width, height = image.size
    left = (width-224)/2
    bottom = (height-224)/2
    right = left + 224
    top = bottom + 224
    
    image = image.crop((left, bottom, right, top))
    image=image.convert('RGB')
    # Normalize
    nimage = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nimage = (nimage - mean) / std
    nimage = nimage.transpose((2, 0, 1))
    return nimage


def validate(model, val_loader, criterion,device):
    val_loss = 0
    accuracy = 0
    if device=='cuda':
       devicee=torch.device("cuda:0")
       model.to(devicee)
    
    model.eval()
    for images, labels in val_loader:

        
        if device=='cuda':
           images, labels = images.to(devicee), labels.to(devicee)
        
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    val_loss=val_loss/len(val_loader)
    accuracy=accuracy/len(val_loader)
    return val_loss, accuracy

def test(model, testloader, criterion,device):
    print("Testing the model on testing dataset .. ")
    test_loss = 0
    accuracy = 0
    if device=='cuda':
       devicee=torch.device("cuda:0")
       model.to(devicee)
    model.eval()

    for images, labels in testloader:

        
        if device=='cuda':
           images, labels = images.to(devicee), labels.to(devicee)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print("Test Loss: {} , Accuracy : {}".format(test_loss/len(testloader),accuracy/len(testloader)))

def save_chk (model,optimizer,epochs,chk_dir,hidden,insize,arch,lrn):
    checkpoint = {'arch': arch,
                  'input_size': insize,
                  'output_size': 102,
                  'hidden_layers': hidden,
                  'model_state_dict': model.state_dict(),
                  'opt_state_dict':optimizer.state_dict(),
                  'epochs':epochs,
                  'learn_rate':lrn
                 }
    
    torch.save(checkpoint, chk_dir)
    print("Model saved.")

def load_chk (filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg':
        model = models.vgg16(pretrained=True)
        
    if  checkpoint['arch'] == 'alexnet'  :
        model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
   
        
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout()),
                          ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))           
                          ]))
    
    epochs=checkpoint['epochs']
    model.classifier=classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer=optim.Adam(model.classifier.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    return model,epochs,optimizer
