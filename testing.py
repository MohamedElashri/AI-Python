def test(model, testloader, criterion,device):
    print("Testing the model on testing dataset .. ")
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()

    for images, labels in testloader:

        images.resize_(images.shape[0], 25088)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print("Test Loss: {} , Accuracy : {}".format(test_loss,accuracy))