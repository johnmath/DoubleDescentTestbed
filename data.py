class Data():
    
    def __init__():
        pass
    



class MNIST(Data):
    
    def __init__():
        self.train_loader = torch.utils.data.DataLoader( 
                                torchvision.datasets.MNIST('./data', 
                                       train=True, 
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                batch_size=batch_size_train, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader( 
                                torchvision.datasets.MNIST('./data', 
                                       train=False, 
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                batch_size=batch_size_train, shuffle=True)
        
        