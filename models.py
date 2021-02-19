import sys
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.optim as optim
import honors_work.data as data
import honors_work.utils as utils
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class TorchModels(nn.Module):
    """This class contains the attributes that all PyTorch models 
    have in common. All PyTorch models will inherit from this class 
    
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
        super(TorchModels, self).__init__()
        
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
        
    
class MultilayerPerceptron(TorchModels):
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
    """
    
    def __init__(self, loss='MSE', dataset='MNIST', cuda=False, optimizer='SGD'):
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
        
        self.param_counts = np.array([1, 2, 3])
        self.current_count = 0
        self.input_layer = nn.Linear(self.data.data_x_dim * self.data.data_y_dim, 
                                     self.param_counts[self.current_count]*10**3)
        self.hidden_layer = nn.Linear(self.param_counts[self.current_count]*10**3, 10)
#         self.mlp_optim = optim.SGD([self.input_layer.weight, self.hidden_layer.weight], lr=.01, 
#                                    momentum=0.95)
#         self.scheduler = optim.lr_scheduler.StepLR(self.mlp_optim, step_size=500, gamma=0.1)
        self.losses = {'train': np.array([]), 'test': np.array([])}
    
    def forward(self, x):
        x = x.view(-1, self.data.data_x_dim * self.data.data_y_dim)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return x
    
    
    
    
class SKLearnModels:
    """This class contains the attributes that all scikit-learn models 
    have in common. All scikit-learn models will inherit from this class
    
    Attributes
    ----------
    N_tree : int
        The number of trees
    N_max_leaves : int
        The maximum number of leaf nodes on a tree
    classifier : RandomForestClassifier
        A scikit-learn random forest model """
    
    def __init__(self, dataset):
        
        data_object = data.SKLearnData()
        data_dict = {'MNIST': data_object.get_mnist}
        X, y, X_val, y_val = data_dict[dataset]()
        self.dataset = {'X': X, 'y': y, 'X_val': X_val, 'y_val': y_val}
        

class RandomForest(SKLearnModels):
    """A Random Forest wrapper that allows for variable numbers of trees 
    and maximum leaf nodes
    
    ...
    Attributes
    ----------
    N_tree : int
        The number of trees
    N_max_leaves : int
        The maximum number of leaf nodes on a tree
    classifier : RandomForestClassifier
        A scikit-learn random forest model 
    """
    
    def __init__(self, dataset='MNIST'):
        
        super(RandomForest, self).__init__(dataset)
        self.N_tree = 1
        self.N_max_leaves = 10
        self.classifier = RandomForestClassifier(n_estimators=self.N_tree, 
                                                 bootstrap=False, 
                                                 criterion='gini', 
                                                 max_leaf_nodes=self.N_max_leaves)
    
    def reinitialize_classifier(self):
        """Helper function for double_descent method"""
        
        self.classifier = RandomForestClassifier(n_estimators=self.N_tree, 
                                                 bootstrap=False, 
                                                 criterion='gini', 
                                                 max_leaf_nodes=self.N_max_leaves)
    
    def double_descent(self):
        """Exhibits double descent in random forest by increasing the
        number of parameters (number of trees and leaf nodes) and training
        each model to completion.
        
        
        ...
        Returns
        -------
        collected_data : dict
            Dictionary of different losses and model attributes collected
            throughout the training process. The keys are {'train_loss', 
            'zero_one_loss', 'mse_loss', 'leaf_sizes', 'trees'} 
        """
        
        leaf_sizes = []
        trees = []
        
        if self.N_tree != 1 or self.N_max_leaves != 10:
            self.N_tree = 1
            self.N_max_leaves = 10
            self.reinitialize_classifier()
        
        training_losses = np.array([])
        zero_one_test_losses = np.array([])
        mse_losses = np.array([])

        while self.N_max_leaves < 2000:

            self.classifier.fit(self.dataset['X'], self.dataset['y'])

            train_loss = utils.sk_zero_one_loss(
                self.classifier.predict(self.dataset['X']), self.dataset['y'])
            
            zero_one_test_loss = utils.sk_zero_one_loss(
                self.classifier.predict(self.dataset['X_val']), self.dataset['y_val'])
            
            mse_loss = utils.sk_mean_squared_error(
                utils.labels_to_vec(self.dataset['y_val']),
                utils.labels_to_vec(self.classifier.predict(self.dataset['X_val'])))

            training_losses = np.append(training_losses, train_loss)
            zero_one_test_losses = np.append(zero_one_test_losses, zero_one_test_loss)
            mse_losses = np.append(mse_losses, mse_loss)

            leaf_sizes.append(self.N_max_leaves)
            trees.append(self.N_tree)
            
            self.N_max_leaves += 100
            self.reinitialize_classifier()

        self.N_max_leaves = self.N_max_leaves - 10
        while self.N_tree <= 20:
            
            self.reinitialize_classifier()
        
            self.classifier.fit(self.dataset['X'], self.dataset['y'])

            train_loss = utils.sk_zero_one_loss(
                self.classifier.predict(self.dataset['X']), self.dataset['y'])
            
            zero_one_test_loss = utils.sk_zero_one_loss(
                self.classifier.predict(self.dataset['X_val']), self.dataset['y_val'])
            
            mse_loss = utils.sk_mean_squared_error(
                utils.labels_to_vec(self.dataset['y_val']),
                utils.labels_to_vec(self.classifier.predict(self.dataset['X_val'])))

            training_losses = np.append(training_losses, train_loss)
            zero_one_test_losses = np.append(zero_one_test_losses, zero_one_test_loss)
            mse_losses = np.append(mse_losses, mse_loss)

            leaf_sizes.append(self.N_max_leaves)
            trees.append(self.N_tree)
            
            self.N_tree += 1
            
            
        return {'train_loss': training_losses, 
                'zero_one_loss': zero_one_test_losses,
                'mse_loss': mse_losses,
                'leaf_sizes': np.array(leaf_sizes), 
                'trees': np.array(trees)}
            
    