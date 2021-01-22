import torch.nn as nn
import torch.optim as optim
import data

class Models:
    """This class contains the attributes that all models have in common.
    
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
        
        self.loss = loss_functions[loss]
        
        # TODO: Add dataset object
        self.dataloaders = dataset

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
    
    def __init__(self, loss='MSE', dataset='MNIST', cuda=False, optim='SGD'):
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
        
        self.hidden = min_param_count
        self.input_layer = nn.Linear(28 * 28,  * 10**3)
        self.hidden_layer = nn.Linear(variable * 10**3, 10)
        self.mlp_optim = optim.SGD
    
    def forward(x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return F.log_softmax(x, dim=1)
        
        
    
    

    
