import sys
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.optim as optim
import honors_work.data as data

class Models:
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
        min_param_count : int
            The starting parameter count (Note: this is x10^3 for this model)
        max_param_count : int
            The upper bound for parameter count (Note: this is x10^3 for this model)
    """
    
    def __init__(self, loss='MSE', dataset='MNIST', cuda=False, optimizer='SGD', min_param_count=1, max_param_count=100):
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
        
        self.min_param_count = min_param_count
        self.max_param_count = max_param_count
        # TODO: Write layer sizes as function of data_dims
        self.input_layer = nn.Linear(28 * 28, 10**3)
        self.hidden_layer = nn.Linear(10**3, 10)
        self.mlp_optim = optim.SGD
    
    def forward(x):
        # TODO: Write view as function of data_dims
        x = x.view(-1, 28 * 28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return F.log_softmax(x, dim=1)
        
        
    
    

    
