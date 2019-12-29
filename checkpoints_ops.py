def save_chk (model,epochs,chk_dir):
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict(),
              'epochs':epochs 
             }
    
    torch.save(checkpoint, chk_dir)

def load_chk (filepath):
    checkpoint = torch.load(filepath)
    model = nn.Sequential(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    epochs=checkpoint['epochs']
    return model,epochs