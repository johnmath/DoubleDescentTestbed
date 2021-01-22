def approx_sigma(mu, max_params):
    """Approximates standard deviation of a normal distribution given 
    the maximum value and mean. In this context, the desired "interpolation 
    threshold" will be the mean, and the desired maximum number of parameters will
    be the maximum value. Using this, we can sample parameter sizes from a normal
    distribution to avoid the computational cost that comes with training over all 
    parameter sizes in some range
    
    Parameters
    ----------
    mu : float
        The chosen value for the interpolation threshold
    max_params : int
        The desired maximum number of parameters to train up to. 
        The meaning of this will vary depending on the model and
        the dataset. 
        (Ex. The value max_params for an MLP training on MNIST would 
        mean max_params * 10^3 parameters will be the maximum number of 
        parameters)
    """
    
    return -(mu - max_params)/3


def train_nn(model, dataloaders, criterion, optimizer, num_epochs=100):
    """Trains a neural network
    
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
    
    train_loss = []
    test_acc = []
    
    dataset_sizes = {'train': len(dataloaders['train'].dataset), 'test': len(dataloaders['test'].dataset) }

    for epoch in range(num_epochs):
            
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        # Switches between training and testing sets
        for phase in ['train', 'test']:
            
            if phase == 'train':
                model.train()
                running_loss = 0.0

            elif phase == 'test':
                model.eval()   # Set model to evaluate mode
                running_test_loss = 0.0

            # Train/Test loop
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                if phase == 'train':
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                        # statistics
                        running_loss += loss.item() * inputs.size(0)

                if phase == 'test':

                    with torch.no_grad():
                        outputs = model(inputs)
                        test_loss = criterion(outputs, labels)
                        running_test_loss += test_loss.item() * inputs.size(0)


#                     if phase == 'train':
#                         scheduler.step()

        train_loss.append(running_loss/ dataset_sizes['train'])
        test_acc.append(running_test_loss/ dataset_sizes['test'])

        print('Train Loss: {:.4f}\nTest Loss {:.4f}'.format(train_loss[epoch], test_acc[epoch]))
            
    return model, train_loss, test_acc

