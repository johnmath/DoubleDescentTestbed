import torch.nn as nn
import torch.optim as optim
import data

class Models:
    """This class contains the attributes that all models have in common.
    
    ...
    
    Attributes
    ----------
        loss : str
            The loss function for the model. Options are {'L1', 'MSE', 
            'CrossEntropy'}.
        dataset : str
            The dataset that the model will be trained on. Options are 
            {'MNIST'}.
        cuda : bool
            If True, the model will train using GPU acceleration if a CUDA
            GPU is available. If False, the model will train on CPU
    """
    
    def __init__(self, loss, dataset, cuda):
        
        loss_functions = {'L1': nn.L1Loss(), 
                          'MSE': nn.MSELoss(), 
                          'CrossEntropy': nn.CrossEntropyLoss()}
        
        self.loss = loss_functions[loss]
        self.dataset = dataset
        self.cuda = cuda
        
    def train():
        pass
        
        
    
class MultilayerPerceptron(Models):
    """A Multilayer Perceptron with a single hidden layer of variable size
    
    ...
    
    Attributes
    ----------
        loss : str
            The loss function for the model. Options are {'L1', 'MSE',
            'CrossEntropy'}.
        dataset : str
            The dataset that the model will be trained on. Options are
            {'MNIST'}.
        cuda : bool
            If True, the model will train using GPU acceleration if a CUDA
            GPU is available. If False, the model will train on CPU
        optim : str
            The optimizer that the model will use while training. Options are
            {'SGD'}
        min
    """
    
    def __init__(self, loss='MSE', dataset='MNIST', cuda=False, optim='SGD'):
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
    
        
        self.mlp_optim = optim.S
    
    

    
