import sys
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.optim as optim
import honors_work.data as data
import honors_work.utils as utils
import torch.nn.functional as F
import numpy as np

class Models(nn.Module):
    """This class contains the attributes that all models have in common.
    All models will inherit from this class 
    
    ...
    Parameters (Not Attributes)
    ---------------------------
        cuda : bool
            If True, the model will train using GPU acceleration if a CUDA
            GPU is available. If False, the model will train on CPU
    
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
        super(Models, self).__init__()
        
        loss_functions = {'L1': nn.L1Loss(), 
                          'MSE': nn.MSELoss(), 
                          'CrossEntropy': nn.CrossEntropyLoss()}
        
        datasets = {'MNIST' : data.MNIST()}
        
        self.loss = loss_functions[loss]
        self.data = datasets[dataset]
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')     
        
    
class MultilayerPerceptron(Models):
    """A Multilayer Perceptron with a single hidden layer of variable size
    
    ...
    Parameters (Not Attributes)
    ---------------------------
        cuda : bool
            If True, the model will train using GPU acceleration if a CUDA
            GPU is available. If False, the model will train on CPU
            
    Attributes
    ----------
        loss : str
            The loss function for the model. Options are {'L1', 'MSE',
            'CrossEntropy'}.
        dataset : str
            The dataset that the model will be trained on. Options are
            {'MNIST'}.
        optim : str
            The optimizer that the model will use while training. Options are
            {'SGD'}    
        param_counts : np.array 
            List of parameter counts that the model will be trained over.
            Since this model is an MLP, these counts correspond to N*10^3 
            neurons for a parameter count, N.
        current_count : int
            The index of the current parameter count in param_counts
        losses : dict
            Dictionary of lists of final losses for each model that 
            is trained at each parameter count
        scheduler : 
    """
    
    def __init__(self, loss='MSE', dataset='MNIST', cuda=False, optimizer='SGD'):
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
        
        self.param_counts = np.array([1, 2, 3])
        self.current_count = 0
        self.input_layer = nn.Linear(self.data.data_x_dim * self.data.data_y_dim, 
                                     self.param_counts[self.current_count]*10**3)
        self.hidden_layer = nn.Linear(self.param_counts[self.current_count]*10**3, 10)
        self.mlp_optim = optim.SGD([self.input_layer.weight, self.hidden_layer.weight], lr=.01, 
                                   momentum=0.95)
        self.scheduler = optim.lr_scheduler.StepLR(self.mlp_optim, step_size=500, gamma=0.1)
        self.losses = {'train': np.array([]), 'test': np.array([])}
    
    def forward(self, x):
        x = x.view(-1, self.data.data_x_dim * self.data.data_y_dim)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return F.log_softmax(x, dim=1)
        

                
        
    
