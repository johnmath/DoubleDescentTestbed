import sys
sys.path.insert(1, '..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import honors_work.data as data
from honors_work.data import utils
from honors_work.data import torch
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
    """A wrapper for a Multilayer Perceptron with a single hidden layer of variable size
    
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
    
    class MLP(nn.Module):
        """TEMP DOCSTRING"""

        def __init__(self, current_count, data, param_counts):
            super(MLP, self).__init__()
            self.data_dims = (data.data_x_dim, data.data_y_dim)

            self.input_layer = nn.Linear(self.data_dims[0] * self.data_dims[1],
                                         param_counts[current_count]*10**3)

            self.hidden_layer = nn.Linear(param_counts[current_count]*10**3, 10)
            
        def forward(self, x):
            x = x.view(-1, self.data_dims[0] * self.data_dims[1])
            x = F.relu(self.input_layer(x))
            x = F.relu(self.hidden_layer(x))
            return x
    
    
    def __init__(self, loss='MSE', 
                 dataset='MNIST', 
                 cuda=False, 
                 optimizer='SGD', 
                 learning_rate=.01, 
                 momentum=.95, 
                 scheduler_step_size=500, 
                 gamma=.1, 
                 current_count=0, 
                 param_counts=np.array([1, 2, 3]),
                 generate_parameters=True):
        
        super(MultilayerPerceptron, self).__init__(loss, dataset, cuda)
        
        self.param_counts = param_counts
        self.current_count = current_count
        self.generate_parameters = generate_parameters
        self.model = self.MLP(self.current_count, self.data, self.param_counts)   
        
        optim_dict = {'SGD': optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)}
        
        self.mlp_optim = optim_dict[optimizer]
        self.scheduler = optim.lr_scheduler.StepLR(self.mlp_optim, step_size=scheduler_step_size, gamma=gamma)
        self.losses = {'train': np.array([]), 'test': np.array([])}
        
        
    @property
    def input_layer(self):
        return classifier.input_layer
    
    
    @property
    def hidden_layer(self):
        return classifier.hidden_layer
    
    
    
    
    def reinitialize_classifier(self):
        new_layer1 = nn.Linear(self.data.data_x_dim * self.data.data_y_dim,
                               self.param_counts[self.current_count]*10**3)
        
        new_layer2 = nn.Linear(self.param_counts[self.current_count]*10**3, 
                               self.data.num_classes)
        new_layer1.weight.data.normal_(0, .01)
        new_layer2.weight.data.normal_(0, .01)

        new_layer1.weight[:10**4] = model.layer1.weight
        new_layer2.weight[:,:10**4] = model.layer2.weight[:]
            
    
    def train(self, model, dataloaders, optimizer, scheduler, num_epochs=100):
        """Trains a neural network

        ...
        Parameters
        ----------
        model
            A PyTorch neural network object (Only strict requirement is that 
            the object has a .forward() method)
        dataloaders : dict
            Dictionary in the format {"train": <train_loader>, "test": <test_loader>}
            where <train_loader> and <test_loader> are PyTorch Dataloaders
        criterion : torch.nn Loss Function
            Loss function that the neural network will use to train and validate the data
        optimizer : torch.optim optimizer
            Optimizer that model will use to minimize the loss
        num_epochs : int
            The number of training epochs that the model will train over. One epoch is
            one full pass through the train and test loaders

        Returns
        -------
        model
             A PyTorch neural network object that has been trained
        train_loss : list
            A list of all training losses at the end of each epoch
        test_acc : list
            A list of all test losses at the end of each epoch
        """

        tb_utils = TensorBoardUtils()

        writer = SummaryWriter('runs/dd_model_{}'.format(model.param_counts[model.current_count]))

        # get some random training images
        dataiter = iter(dataloaders['train'])
        images, labels = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        tb_utils.matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('MNIST Dataset', img_grid)
        writer.add_graph(model, images)
        writer.close()

        train_loss = []
        test_acc = []

        dataset_sizes = {'train': len(dataloaders['train'].dataset), 'test': len(dataloaders['test'].dataset) }

        model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.95)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        print('Model with parameter count {}'.format(model.param_counts[model.current_count]))
        print('-' * 10)

        for epoch in range(num_epochs):

            #print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            # print('-' * 10)

            # Switches between training and testing sets
            for phase in ['train', 'test']:

                if phase == 'train':
                    model.train()
                    running_loss = 0.0
                elif phase == 'test':
                    model.eval()   # Set model to evaluate mode
                    running_test_loss = 0.0

                # Train/Test loop
                for i, d in enumerate(dataloaders[phase], 0):

                    inputs, labels = d

                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()

                    if phase == 'train':
                        outputs = model.forward(inputs)
                        loss = model.loss(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                        # statistics
                        running_loss += loss.item() * inputs.size(0)

                    if phase == 'test':
                        outputs = model.forward(inputs)
                        test_loss = model.loss(outputs, labels)
                        running_test_loss += test_loss.item() * inputs.size(0)

                    if i % 50 == 49:    # every 1000 mini-batches...

                        # ...log the running loss

                        # function: add_scalars
                        writer.add_scalar('Training Loss',
                                        running_loss / 1000,
                                        epoch * len(dataloaders['train']) + i)

                        # ...log a Matplotlib Figure showing the model's predictions on a
                        # random mini-batch
                        writer.add_figure('Predictions vs. Actuals',
                                        tb_utils.plot_classes_preds(model, inputs, labels),
                                        global_step=epoch * len(dataloaders['train']) + i)

                if phase == 'train':
                    scheduler.step()

            train_loss.append(running_loss/ dataset_sizes['train'])        
            test_acc.append(running_test_loss/ dataset_sizes['test'])

            if train_loss[-1] < 10**-5:
                break

        print('Train Loss: {:.4f}\nTest Loss {:.4f}'.format(train_loss[-1], test_acc[-1]))

        return model, train_loss, test_acc
    
    
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
        
        # Run if user wants to use a manually created parameter count list

            
            
        while self.current_count < len(self.param_counts):

            _, train_loss, test_loss = self.train(self.classifier)

            self.losses['train'] = np.append(self.losses['train'], train_loss[-1])
            self.losses['test'] = np.append(self.losses['test'], test_loss[-1])

            self.current_count += 1
            if self.current_count < len(self.param_counts):
                self.reinitialize_classifier()

            np.save('train_loss.npy', self.losses['train'])
            np.save('test_loss.npy', self.losses['test'])
            np.save('parameter_counts', self.param_counts)
                
        if not self.generate_parameters:
            
            return {'train_loss': self.losses['train'],
                    'test_loss': self.losses['test'],
                    'parameter_counts': self.param_counts}
        
        
        self.current_count -= 1
        flag = False
        post_flag = 0

        while post_flag < 4:

            next_ct, flag = utils.get_next_param_count(self.param_counts, 
                                                 self.losses['test']/self.losses['test'].sum(), 
                                                 flag)

            self.param_counts = np.append(self.param_counts, next_ct)
            self.current_count += 1
            self.reinitialize_classifier()

            _, train_loss, test_loss = self.train(self.classifier)

            self.losses['train'] = np.append(self.losses['train'], train_loss[-1])
            self.losses['test'] = np.append(self.losses['test'], test_loss[-1])

            if flag and (self.param_counts[-1] - self.param_counts[-2]) != 1:
                post_flag += 1

            np.save('train_loss.npy', self.losses['train'])
            np.save('test_loss.npy', self.losses['test'])
            np.save('parameter_counts', self.param_counts)
        
        return {'train_loss': self.losses['train'],
                'test_loss': self.losses['test'],
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
    
    def __init__(self, dataset):
        
        data_object = data.SKLearnData()
        data_dict = {'MNIST': data_object.get_mnist}
        X, y, X_val, y_val = data_dict[dataset](samples=6000)
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
    
    def __init__(self, dataset='MNIST'):
        
        super(RandomForest, self).__init__(dataset)
        self.N_tree = 1
        self.N_max_leaves = 10
        print('Initializing RandomForest')
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
            
    