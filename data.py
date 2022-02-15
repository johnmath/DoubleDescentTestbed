import torch
import DoubleDescentTestbed.utils as utils
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
import os

class TorchData:
    """This class contains the attributes that all datasets have in common.
    All datasets will inherit from this class
    
    ...
    Attributes
    ----------
    train_loader : PyTorch Dataloader
        The dataloader for the training set
    train_loader : PyTorch Dataloader
        The dataloader for the testing set
    data_x_dim : int
        The size of the x-dimension for each image in the dataset
    data_y_dim : int
        The size of the y-dimension for each image in the dataset  
    """
    
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.data_x_dim = None
        self.data_y_dim = None

class MNIST(TorchData):
    """The MNIST Dataset (Handwritten Digits)
    
    ...
    Attributes
    ----------
    train_loader : PyTorch Dataloader
        The dataloader for the training set
    train_loader : PyTorch Dataloader
        The dataloader for the testing set
    data_x_dim : int
        The size of the x-dimension for each image in the dataset
    data_y_dim : int
        The size of the y-dimension for each image in the dataset
    train_batch_size : int
        The number of training examples per batch
    test_batch_size : int
        The number of testing examples per batch
    dataloaders : dict
        A dictionary that contains the 2 dataloaders. The keys are 
        "train" and "test"
    """
    
    def __init__(self, training_samples=4000, train_batch=128, test_batch=128):
        print('Initializing MNIST')
        self.train_batch_size = train_batch
        self.test_batch_size = test_batch
        self.training_samples = training_samples
        
        self.train_loader = torch.utils.data.DataLoader( 
                                torchvision.datasets.MNIST('./data', 
                                   train=True, 
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                batch_size=self.train_batch_size, shuffle=True)
        
        train_dataset = torch.utils.data.Subset(self.train_loader.dataset,
                                                range(0, training_samples))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.train_batch_size, 
                                                        shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader( 
                        torchvision.datasets.MNIST('./data', 
                           train=False, 
                           download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                        batch_size=self.test_batch_size, shuffle=False)
        
        test_dataset = torch.utils.data.Subset(self.test_loader.dataset, 
                                               range(0, int(training_samples//.75) - training_samples))
        
        self.test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=self.test_batch_size,
                                                       shuffle=True)
        
        # Only use a subset of the MNIST dataset for MLP
        self.dataloaders = {'train': self.train_loader,
                            'test': self.test_loader}
        
        self.data_x_dim = self.train_loader.dataset[0][0].shape[1]
        self.data_y_dim = self.train_loader.dataset[0][0].shape[2]
        self.classes = list(set([self.dataloaders['train'].dataset[i][1] for i in range(len(self.dataloaders['train']))]))
        self.num_classes = len(self.classes)
        self.samples = len(self.train_loader.dataset)
        self.dataset_sizes = {'train': self.samples, 'test': len(self.test_loader.dataset)}

class SKLearnData:
    
    def __init__(self):
        pass
    
    def get_mnist(filename=None, samples=100):
        """Returns a subset of the the MNIST Dataset as numpy arrays
        
        ...
        Parameters
        ----------
        samples : int
            The number of datapoints that the user wants to be 
            returned. The size of the returned validation set 
            will be samples/2
        filename : str
            The filename that the dataset will be saved to.

        Returns
        -------
        X : np.array
            Training set of 784 (28*28) dimensional vectors 
            that correspond to 28x28 MNIST images
        y : np.array
            Labels for each of the vectors in X
        X_val : np.array
            Training set of 784 (28*28) dimensional vectors 
            that correspond to 28x28 MNIST images
        y_val : np.array
            Labels for each of the vectors in X_val
        """

        
        if os.path.exists('scikit_learn_data/X_saved.npy') and os.path.exists('scikit_learn_data/Y_saved.npy'):    
            X = np.load('scikit_learn_data/X_saved.npy')
            y = np.load('scikit_learn_data/Y_saved.npy', allow_pickle=True)
        else:
            print('Fetching MNIST')
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            np.save('scikit_learn_data/X_saved.npy', X)
            np.save('scikit_learn_data/Y_saved.npy', y)
        
        if (samples + samples//2) > X.shape[0] - 1:
            samples = X.shape[0] - 1
        
        X_val = X[samples + 1:(samples + samples//2)]
        y_val = y[samples + 1:(samples + samples//2)]
        X = X[:samples + 1]
        y = y[:samples + 1]
        
        return X, y, X_val, y_val
        
        