from scipy.stats import norm 
import numpy as np

def approx_sigma(mu, max_params):
    """Approximates standard deviation of a normal distribution given 
    the maximum value and mean. In this context, the desired "interpolation 
    threshold" will be the mean, and the desired maximum number of parameters will
    be the maximum value. Using this, we can sample parameter sizes from a normal
    distribution to avoid the computational cost that comes with training over all 
    parameter sizes in some range
    
    The standard deviation is calculated using the fact that 99.7% of the elements
    in a normal distribution fall within 3*sigma of the mean. So, we can use this
    to approximate the maximum value using max = mean - 3*sigma. Therefore, a rough
    approximate for the standard deviation would be sigma = (max - mean)/3.
    
    Parameters
    ----------
    mu : int
        The chosen value for the interpolation threshold
    max_params : int
        The desired maximum number of parameters to train up to. 
        The meaning of this will vary depending on the model and
        the dataset. 
        (Ex. The value max_params for an MLP training on MNIST would 
        mean max_params * 10^3 parameters will be the maximum number of 
        parameters)
        
        
    Returns
    -------
    std : float
        Standard deviation of a normal distribution
    """
    
    return (max_params - mu)/3


def get_parameter_counts(mu, max_params, num_samples):
    """Generate a list of paramter counts using a normal distribution
    centered around the parameter count that signifies the interpolation
    threshold
    
    Parameters
    ----------
    mu : int
        The chosen value for the interpolation threshold
    max_params : int
        The desired maximum number of parameters to train up to. 
        The meaning of this will vary depending on the model and
        the dataset. 
        (Ex. The value max_params for an MLP training on MNIST would 
        mean max_params * 10^3 parameters will be the maximum number of 
        parameters)
    num_samples : int
        Desired number of parameter counts to be returned. (Note: 
        The number of parameter counts returned will not be exactly
        be num_samples because parameter counts must be integers, 
        and we are not allowing for duplicate parameter counts. 
        It will usually be between 3/4*num_samples and num_samples.
    
    Returns
    -------
    paramater_counts : list
        A sorted list of parameter counts to train over
    """
    
    sigma_est = approx_sigma(mu, max_params)
    
    # How many sigmas away from the mean is 0?
    # Use mu - w*sigma = 0
    # -> w = mu/sigma
    w = mu/sigma_est
    
    # What percentile is 0 in our distribution?
    # P[X <= mu - w*sigma] since mu - w*sigma = 0
    # -> P[X <= Z] where Z = ((mu - w*sigma) - mu)/sigma
    # So, Z = -w
    percentile = norm.cdf(-w)
    print(percentile)
    
    # Use this percentile to compensate for the amount of 
    # parameter_counts we lose when we turn sampels into ints 
    # and threshold > 0
    compensated_num_samples = int(round((1 + percentile)*num_samples, 0))
    dist = np.random.normal(loc=mu, scale=sigma_est, size=compensated_num_samples)
    dist = dist[dist>=0]
    parameter_counts = [int(round(x, 0)) for x in dist]
    return sorted(list(set(parameter_counts)))


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

