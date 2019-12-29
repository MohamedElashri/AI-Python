import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args(src):
    
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    if src=='train' : 
        parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                    help = 'path to the folder of training images') 
        parser.add_argument('--arch', type = str, default = 'resnet',choices=['alexnet','vgg'],
                    help = 'Select one of the three CNN models impelemented in project') 
        parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                    help = 'path to save chackpoint of the trained model') 
        parser.add_argument('--learning_rate', type = float, default = 0.01, 
                    help = 'Learning rate of the model') 
        parser.add_argument('--hidden_units', type = int, default = 4096, 
                    help = 'number of the hidden units of the nerual network') 
        parser.add_argument('--epochs', type = int, default = 5, 
                    help = 'Number of epochs of the training') 
        parser.add_argument('--device', type = str, default = 'cpu',choices=['cuda', 'cpu'], 
                    help = 'Device used in training') 
    
    else : 
        parser.add_argument('--device', type = str, default = 'cpu',choices=['cuda', 'cpu'], 
                    help = 'Device used in training') 
        parser.add_argument('--input_dir', type = str, 
                    help = 'path to the image  to predict')
        parser.add_argument('--chk_dir', type = str, default = 'checkpoint.pth', 
                    help = 'path to load chackpoint of the trained model') 
        parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'path to map classes names to its indexes') 
        parser.add_argument('--top_k', type = int, default = 5, 
                    help = 'Learning rate of the model') 
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    # Replace None with parser.parse_args() parsed argument collection that 
    args = parser.parse_args()
    # you created with this function 
    return args
