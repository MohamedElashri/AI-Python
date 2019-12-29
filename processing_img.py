from PIL import Image
def process_image(imagedir):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image=Image.open(imagedir)
    imgae=image.thumbnail((256,256), resample=3)
    image=image.crop((16,16,240,240))
    # Resize
    if image.size[0] > image.size[1]:
        image=image.thumbnail((image.size[0], 256))
    else:
        image=image.thumbnail((256, image.size[1]))
        
    
    # Crop 
    left = (image.width-224)/2
    bottom = (image.height-224)/2
    right = left_margin + 224
    top = bottom_margin + 224
    
    image = image.crop((left, bottom, right, top))
    # Normalize
    nimage = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nimage = (nimage - mean) / std
    nimage = nimage.transpose((2, 0, 1))
    return nimage