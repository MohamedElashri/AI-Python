def validate(model, val_loader, criterion,device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    for images, labels in val_loader:

        images.resize_(images.shape[0], 25088)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print("Loss: {} , Accuracy : {}".format(val_loss,accuracy))
