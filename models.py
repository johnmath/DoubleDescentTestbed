import sys
sys.path.insert(1, '..')
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import DoubleDescentTestbed.data as data
from DoubleDescentTestbed.data import utils
from DoubleDescentTestbed.utils import TensorBoardUtils, torchvision, np
from DoubleDescentTestbed.data import torch
import torch.nn.functional as F
import os, shutil
from sklearn.ensemble import RandomForestClassifier


class TorchModels():
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
        training_samples : int
            Desired number of elements from the training set
    """
    
    def __init__(self, loss, dataset, batch_size, training_samples, cuda):
        super(TorchModels, self).__init__()
        
        loss_functions = {'L1': nn.L1Loss(), 
                          'MSE': nn.MSELoss(), 
                          'CrossEntropy': nn.CrossEntropyLoss()}
        
        datasets = {'MNIST' : data.MNIST(training_samples=training_samples, train_batch=batch_size, test_batch=batch_size)}
        
        self.loss = loss_functions[loss]
        self.data = datasets[dataset]
        self.cuda = cuda
        
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')     
        
    
class MultilayerPerceptron(TorchModels):
    """A wrapper for a multilayer perceptron with a single hidden layer of variable size
    
    ...
    
    Attributes
    ----------
        
        loss : str
            The loss function for the model. Options are {'L1', 'MSE', 
            'CrossEntropy'}.  
        dataset : str
            The dataset that the model will be trained on. Options are
            {'MNIST'}.
        batch_size : int
            The batch size for the training set
        cuda : bool
            If True, cuda will be used instead of cpu
        optimizer : str
            The optimizer that the model will use while training. Options are
            {'SGD'}
        learning_rate : float
            Learning rate for optimizer
        momentum : float
            Momentum parameter to accelerate SGD
        scheduler_step_size : int
            Number of iterations before applying learning rate scheduler
        gamma : float
            Learning rate scheduler factor
        current_count : int
            The index of the current parameter count in param_counts
        param_counts : np.array 
            List of parameter counts that the model will be trained over.
            Since this model is an MLP, these counts correspond to N*10^3 
            neurons for a parameter count, N.
        generate_parameters : True
            Uses a parameter adaptation algorithm to predict the next best model 
            to train by analyzing the final loss vs hidden layer size of all previous
            models
        training_samples : int
            Desired number of elements from the training set
        factor : int
            Multiplier for param_counts. factor * param_counts[i] = number
            of neurons in hidden layer
        reuse_weights : True
            If True, reuses weights from previous model to help next model converge
            more quickly
        seed : int
            Seed for random weight initialization
        max_epochs : int
            The max number of iterations to train each model
    """
    
    class MLP(nn.Module):
        """An implementation of a 2-layer multilayer perceptron that allows 
        for changing the number of neurons in the hidden layer"""
        
        def __init__(self, current_count, data, param_counts, factor, hidden_layer_size):
            super().__init__()
            print(f'Initializing MLP with {hidden_layer_size} hidden units')

            self.data_dims = (data.data_x_dim, data.data_y_dim)
            
            self.input_layer = nn.Linear(self.data_dims[0] * self.data_dims[1],
                                         hidden_layer_size)

            self.hidden_layer = nn.Linear(hidden_layer_size, data.num_classes)
            
        def forward(self, x):
            x = x.view(-1, self.data_dims[0] * self.data_dims[1])
            x = F.relu(self.input_layer(x))
            x = self.hidden_layer(x)
            return x
    
    
    def __init__(self, loss='CrossEntropy', 
                 dataset='MNIST',
                 batch_size=128,
                 cuda=False, 
                 optimizer='SGD', 
                 learning_rate=.01, 
                 momentum=.95, 
                 scheduler_step_size=500, 
                 gamma=.9, 
                 current_count=0, 
                 param_counts=np.array([1, 2, 3]),
                 generate_parameters=True,
                 training_samples=4000,
                 factor=10**3,
                 reuse_weights=True,
                 seed=None,
                 max_epochs=1000):
        
        super(MultilayerPerceptron, self).__init__(loss, dataset, batch_size, training_samples, cuda)
        
        if seed:
            torch.manual_seed(seed)
        self.param_counts = param_counts
        self.current_count = current_count
        self.samples = training_samples
        self.post_flag = 0
        self.generate_parameters = generate_parameters
        self.factor = factor
        self.reuse_weights = reuse_weights
        self.model = self.MLP(self.current_count, self.data, self.param_counts, self.factor, self.hidden_layer_size)  
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optim_dict = {'SGD': optim.SGD(self.model.parameters(),
                                            lr=self.learning_rate,
                                            momentum=self.momentum)}
        
        self.optimizer = optimizer
        self.mlp_optim = self.optim_dict[self.optimizer]
        self.gamma = gamma 
        self.scheduler_step_size = scheduler_step_size
        self.scheduler = optim.lr_scheduler.StepLR(self.mlp_optim, 
                                                   step_size=self.scheduler_step_size, 
                                                   gamma=self.gamma)
        
        self.losses = {'train': np.array([]), 
                       'test': np.array([]), 
                       'zero_one_train': np.array([]), 
                       'zero_one_test': np.array([])}
        self.max_epochs = max_epochs
        
        
    @property
    def input_layer(self):
        return self.model.input_layer
    
    
    @property
    def hidden_layer(self):
        return self.model.hidden_layer
    
    @property 
    def hidden_layer_size(self):
        # This computes the size of the hidden layer, H, using the equation
        # Total_Parameters = (d+1)*H + (H + 1)*K
        return (self.param_counts[self.current_count] * self.factor \
                + self.data.num_classes)//(self.data.data_x_dim * self.data.data_y_dim + 1)
    
    def reinitialize_classifier(self):
        """Uses new parameter count to initialize the next MLP.
        The N weights from the previous model are transplanted into 
        the first N spots of the new model with M > N parameters"""
        
        new_model = self.MLP(self.current_count, self.data, self.param_counts, self.factor, self.hidden_layer_size)
                
        if self.reuse_weights:
                 
            in_weights = torch.randn_like(new_model.input_layer.weight)*.01
            hidden_weights = torch.randn_like(new_model.hidden_layer.weight)*.01
                
            in_weights[:self.model.input_layer.weight.shape[0]] = self.model.input_layer.weight
            hidden_weights[:,:self.model.hidden_layer.weight.shape[1]] = self.model.hidden_layer.weight[:]
            
            new_model.input_layer.weights = torch.nn.Parameter(data=in_weights)
            new_model.hidden_layer.weights = torch.nn.Parameter(data=hidden_weights)
        
        self.model = new_model
        
        self.optim_dict = {'SGD': optim.SGD(self.model.parameters(),
                                            lr=self.learning_rate,
                                            momentum=self.momentum)}
        
        self.mlp_optim = self.optim_dict[self.optimizer]
        
        if self.param_counts[self.current_count] * self.factor > self.samples * self.data.num_classes:
            self.gamma = 1
            
        self.scheduler = optim.lr_scheduler.StepLR(self.mlp_optim, 
                                                   step_size=self.scheduler_step_size, 
                                                   gamma=self.gamma) 
    
    
    def train(self):
        """Trains the MLP model using the selected loss function,
        optimizer, and scheduler. This also outputs to tensorboard.
        To access all of the summaries for trained models, run the 
        tensorboard command in another command line while the model 
        is training

        ...
        
        Returns
        -------
        model
             A PyTorch neural network object that has been trained
        train_loss : list
            A list of all training losses at the end of each epoch
        test_acc : list
            A list of all test losses at the end of each epoch
        zero_one_loss : list 
            A list of all 0-1 training losses at the end of each epoch
        zero_one_acc : list 
            A list of all 0-1 test losses at the end of each epoch
        """
        
        tb_utils = TensorBoardUtils()
        model_writer = SummaryWriter(f'mlp-runs/dd_model_{self.param_counts[self.current_count]}')

        # get some random training images
        dataiter = iter(self.data.dataloaders['train'])
        images, labels = dataiter.next()
        
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        tb_utils.matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        model_writer.add_image('MNIST Dataset', img_grid)
        model_writer.add_graph(self.model, images)
        model_writer.close()

        train_loss = []
        test_acc = []
        zero_one_loss = []
        zero_one_acc = []

        print('Model with parameter count {}'.format(self.param_counts[self.current_count]))
        print('-' * 10)

        if self.cuda:
            self.model = self.model.cuda()
                
        for epoch in range(self.max_epochs):
            #print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            # print('-' * 10)

            # Switches between training and testing sets
            for phase in ['train', 'test']:

                if phase == 'train':
                    self.model.train()
                    running_loss = 0.0
                    running_zero_one_loss = 0.0
                elif phase == 'test':
                    self.model.eval()   # Set model to evaluate mode
                    running_test_loss = 0.0
                    running_zero_one_acc = 0.0

                # Train/Test loop
                for i, d in enumerate(self.data.dataloaders[phase], 0):

                    inputs, labels = d
                    
                    if self.cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    self.mlp_optim.zero_grad()
                                        
                    if phase == 'train':
                        outputs = self.model.forward(inputs)
                        loss = self.loss(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        self.mlp_optim.step()
                        zero_one_train = utils.torch_zero_one_loss(outputs, labels)
                        running_zero_one_loss += zero_one_train.item() * inputs.size(0)
                        running_loss += loss.item() * inputs.size(0)

                    if phase == 'test':
                        outputs = self.model.forward(inputs)
                        test_loss = self.loss(outputs, labels)
                        zero_one_test = utils.torch_zero_one_loss(outputs, labels)
                        running_zero_one_acc += zero_one_test.item() * inputs.size(0)
                        running_test_loss += test_loss.item() * inputs.size(0)
                        
                if phase == 'train' and self.post_flag == False:
                    self.scheduler.step()

            train_loss.append(running_loss/ self.data.dataset_sizes['train'])        
            test_acc.append(running_test_loss/ self.data.dataset_sizes['test'])
            zero_one_loss.append(running_zero_one_loss/self.data.dataset_sizes['train'])
            zero_one_acc.append(running_zero_one_acc/self.data.dataset_sizes['test'])
            
            model_writer.add_scalar(f'Train-Loss/{self.hidden_layer_size} hidden units; {self.param_counts[self.current_count]}*{self.factor} total parameters',
                              train_loss[-1],
                             epoch)
            
            model_writer.add_scalar(f'Test-Loss/{self.hidden_layer_size} hidden units; {self.param_counts[self.current_count]}*{self.factor} total parameters',
                              test_acc[-1],
                              epoch)

            model_writer.add_scalar(f'Train-Loss/{self.hidden_layer_size} hidden units; {self.param_counts[self.current_count]}*{self.factor} total parameters (Zero-One)',
                                    zero_one_loss[-1],
                                    epoch)
            
            model_writer.add_scalar(f'Test-Loss/{self.hidden_layer_size} hidden units; {self.param_counts[self.current_count]}*{self.factor} total parameters (Zero-One)',
                                    zero_one_acc[-1],
                                    epoch)
            
            
            if (zero_one_loss[-1] == 0 or train_loss[-1] < 10**-5):
                if self.generate_parameters:
                    if self.post_flag:
                        break
                        
                if self.param_counts[self.current_count] * self.factor < self.samples * self.data.num_classes:
                    break

        print('Train Loss: {:.4f}\nTest Loss {:.4f}\n{} Hidden Units'.format(train_loss[-1], test_acc[-1], self.hidden_layer_size))
        
        torch.cuda.empty_cache()
        
        return self.model, train_loss, test_acc, zero_one_loss, zero_one_acc
    
    
    def double_descent(self):
        """Uses the train and get_next_param_count methods 
        to train the same architecture with varying parameter
        sizes. This method also keeps track of the final losses 
        of each model that is trained

        ...
        Parameters
        ----------
        model : Models instance
            The model object that will be trained with varying 
            parameter sizes

        Returns
        -------
        None
        """
        try:
            os.makedirs('mlp-output')
        except Exception as E:
            print('Could not make mlp-output')
            print(E)
            pass
        
        try:
            shutil.rmtree('mlp-runs')
        except:
            pass
        
        dd_writer = SummaryWriter('mlp-runs/double-descent')
        while self.current_count < len(self.param_counts):

            _, train_loss, test_loss, zero_one_train, zero_one_test = self.train()

            self.losses['train'] = np.append(self.losses['train'], train_loss[-1])
            self.losses['test'] = np.append(self.losses['test'], test_loss[-1])
            self.losses['zero_one_train'] = np.append(self.losses['zero_one_train'], zero_one_train[-1])
            self.losses['zero_one_test'] = np.append(self.losses['zero_one_test'], zero_one_test[-1])
            
            dd_writer.add_scalar('MLP-Double-Descent/Train', 
                                 self.losses['train'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Test', 
                                 self.losses['test'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Train (Zero-One)', 
                                 self.losses['zero_one_train'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Test (Zero-One)', 
                                 self.losses['zero_one_test'][-1], 
                                 self.param_counts[self.current_count])
            self.current_count += 1
            
            if self.current_count < len(self.param_counts):
                self.reinitialize_classifier()
        
            np.save('mlp-output/train_loss.npy', self.losses['train'])
            np.save('mlp-output/test_loss.npy', self.losses['test'])
            np.save('mlp-output/zero_one_train.npy', self.losses['zero_one_train'])
            np.save('mlp-output/zero_one_test.npy', self.losses['zero_one_test'])
            np.save('mlp-output/parameter_counts', self.param_counts)
                
        if not self.generate_parameters:
            
            return {'train_loss': self.losses['train'],
                    'test_loss': self.losses['test'],
                    'zero_one_train': self.losses['zero_one_train'],
                    'zero_one_test': self.losses['zero_one_test'],
                    'parameter_counts': self.param_counts}
        
        
        self.current_count -= 1
        flag = False
        while self.post_flag < 4:
            

            next_ct, flag = utils.get_next_param_count(self.param_counts, 
                                                 self.losses['test']/self.losses['test'].sum(), 
                                                 flag)

            self.param_counts = np.append(self.param_counts, next_ct)
            self.current_count += 1
            self.reinitialize_classifier()

            _, train_loss, test_loss, zero_one_train, zero_one_test = self.train()

            self.losses['train'] = np.append(self.losses['train'], train_loss[-1])
            self.losses['test'] = np.append(self.losses['test'], test_loss[-1])
            self.losses['zero_one_train'] = np.append(self.losses['zero_one_train'], zero_one_train[-1])
            self.losses['zero_one_test'] = np.append(self.losses['zero_one_test'], zero_one_test[-1])
            
            dd_writer.add_scalar('MLP-Double-Descent/Train', 
                                 self.losses['train'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Test', 
                                 self.losses['test'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Train (Zero-One)', 
                                 self.losses['zero_one_train'][-1], 
                                 self.param_counts[self.current_count])
            dd_writer.add_scalar('MLP-Double-Descent/Test (Zero-One)', 
                                 self.losses['zero_one_test'][-1], 
                                 self.param_counts[self.current_count])

            if flag and (self.param_counts[-1] - self.param_counts[-2]) != 1:
                print('Iterating Post Flag')
                self.post_flag += 1
                print(f'Post Flag {self.post_flag}')

            np.save('mlp-output/train_loss.npy', self.losses['train'])
            np.save('mlp-output/test_loss.npy', self.losses['test'])
            np.save('mlp-output/zero_one_train.npy', self.losses['zero_one_train'])
            np.save('mlp-output/zero_one_test.npy', self.losses['zero_one_test'])
            np.save('mlp-output/parameter_counts', self.param_counts)
        
        return {'train_loss': self.losses['train'],
                'test_loss': self.losses['test'],
                'zero_one_train': self.losses['zero_one_train'],
                'zero_one_test': self.losses['zero_one_test'],
                'parameter_counts': self.param_counts}
    
    

    
    
class SKLearnModels:
    """This class contains the attributes that all scikit-learn models 
    have in common. All scikit-learn models will inherit from this class
    
    ...
    Parameters (Not Attributes)
    ---------------------------
    dataset : str
        A string that represents the dataset that the user wants to train
        the model on. The current list is {MNIST}
    
    Attributes
    ----------
    dataset : np.array
        The chosen dataset from the list {MNIST}
    """
    
    def __init__(self, dataset, samples):
        
        data_object = data.SKLearnData()
        data_dict = {'MNIST': data_object.get_mnist}
        X, y, X_val, y_val = data_dict[dataset](samples=samples)
        self.dataset = {'X': X, 'y': y, 'X_val': X_val, 'y_val': y_val}
        

class RandomForest(SKLearnModels):
    """A Random Forest wrapper that allows for variable numbers of trees 
    and maximum leaf nodes
    
    ...
    Parameters (Not Attributes)
    ---------------------------
    dataset : str
        A string that represents the dataset that the user wants to train
        the model on. The current list is {MNIST}
    
    Attributes
    ----------
    N_tree : int
        The number of trees
    N_max_leaves : int
        The maximum number of leaf nodes on a tree
    classifier : RandomForestClassifier
        A scikit-learn random forest model 
    """
    
    def __init__(self, dataset='MNIST', 
                 N_tree=1, 
                 N_max_leaves=10, 
                 bootstrap=False, 
                 criterion='gini', 
                 samples=4000, 
                 leaves_limit=2000,
                 tree_limit=20,
                 leaves_iter=100,
                 tree_iter=1):
        
        self.dataset_name = dataset
        super(RandomForest, self).__init__(dataset, samples)
        self.N_tree = N_tree
        self.samples = samples
        self.N_max_leaves = N_max_leaves
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.leaves_limit = leaves_limit
        self.tree_limit = tree_limit
        self.leaves_iter = leaves_iter
        self.tree_iter = tree_iter
        print('Initializing RandomForest')
        self.classifier = RandomForestClassifier(n_estimators=self.N_tree, 
                                                 bootstrap=self.bootstrap, 
                                                 criterion=self.criterion, 
                                                 max_leaf_nodes=self.N_max_leaves)
    
    def reinitialize_classifier(self):
        """Helper function for double_descent method"""
        
        self.classifier = RandomForestClassifier(n_estimators=self.N_tree, 
                                                 bootstrap=self.bootstrap, 
                                                 criterion=self.criterion, 
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
        
        training_losses = np.array([])
        zero_one_test_losses = np.array([])
        mse_losses = np.array([])

        while self.N_max_leaves < self.leaves_limit:

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
            
            self.N_max_leaves += self.leaves_iter
            self.reinitialize_classifier()

        self.N_max_leaves = self.N_max_leaves - self.leaves_iter
        while self.N_tree <= self.tree_limit:
            
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
            
            self.N_tree += self.tree_iter
            
            
        return {'train_loss': training_losses, 
                'zero_one_loss': zero_one_test_losses,
                'mse_loss': mse_losses,
                'leaf_sizes': np.array(leaf_sizes), 
                'trees': np.array(trees),
                'samples': self.samples,
                'dataset': self.dataset_name}
            
    