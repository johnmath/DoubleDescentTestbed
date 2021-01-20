import torch.nn as nn
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
    
    def __init__(self, loss, dataset):
        
        loss_functions = {'L1': nn.L1Loss(), 
                          'MSE': nn.MSELoss(), 
                          'CrossEntropy': nn.CrossEntropyLoss()}
        
        self.loss = loss_functions[loss]
        self.dataset = dataset
        
        
    
class MultilayerPerceptron(Models):
    """A Multilayer Perceptron with a single hidden layer of variable size
    
    ...
    
    Attributes
    ----------
        mlp_loss : str
            The loss function for the model. Options are {'L1', 'MSE',
            'CrossEntropy'}.
        mlp_dataset : str
            The dataset that the model will be trained on. Options are
            {'MNIST'}.
        cuda : bool
            If True, the model will train using GPU acceleration if a CUDA
            GPU is available. If False, the model will train on CPU
    """
    
    def __init__(self, mlp_loss='MSE', mlp_dataset='MNIST'):
        super(MultilayerPerceptron, self).__init__(mlp_loss, mlp_dataset)
        