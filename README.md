# Double Descent Testbed

## Getting Started

The Double Descent Testbed is an open-source platform to conduct experiments related to the [double descent phenomenon](https://openai.com/blog/deep-double-descent/). Read the full blog post here:

http://www.johnabascal.com/machine/learning/2022/02/14/double-descent-testbed.html

### Before you start

You should have [Anaconda](https://www.anaconda.com/products/individual) installed. This is the package manager we will use to install all of the dependencies.

Once Anaconda is installed, install [PyTorch](https://pytorch.org/get-started/locally/) (along with torchvision). If you have access to an Nvidia GPU, make sure to install PyTorch with CUDA support. GPU acceleration will *greatly* reduce the amount of time to conduct experiments. You will also need to install [NumPy](https://numpy.org/install/), [scikit-learn](https://scikit-learn.org/stable/install.html), [TensorBoard](https://anaconda.org/conda-forge/tensorboardand), and [matplotlib](https://matplotlib.org/stable/users/installing/index.html)

### Running your first experiment

It only takes a few lines of code to run an entire double descent experiment using the testbed.

First, we have to import our machine learning model of interest. In this case, we will use the Multilayer Perceptron. We will also import `numpy` and `pickle` and create functions to save and load the results from this experiment as dictionaries. This way, we do not have to run time-consuming processes, such as training loops, multiple times.

```python
from DoubleDescentTestbed.models import MultilayerPerceptron
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
```

Now, we can initialize the model and enter the desired parameters for our experiment. These parameters include the training set, the loss function, the number of parameters for each model, etc. In this case, we are not using the parameter counts generation algorithm, and we are reusing weights from previous models to speed up the convergence of larger models. The default dataset (MNIST) will be used, and the number of parameters for each model will be `param_counts[i]*(10^3)`.


```python
model = MultilayerPerceptron(cuda=True, 
                             loss='CrossEntropy',
                             param_counts=np.array([3, 4, 7, 10, 15, 20, 23, 27, 31, 32, 33, 34, 36, 38, 40, 41, 42, 60, 100, 150, 300, 800]),
                             generate_parameters=False,
                             max_epochs=6000,
                             batch_size=128,
                             seed=0,
                             reuse_weights=False)

outs = model.double_descent()
```

Throughout training, statistics are aggregated. At the end of the experiment, we are given an output that we can save using the functions we wrote after the import statements.

```python
save_obj(outs, 'mlp-dd-experiment-no-reuse')
```