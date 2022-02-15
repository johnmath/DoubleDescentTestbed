from DoubleDescentTestbed.models import MultilayerPerceptron
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


model = MultilayerPerceptron(cuda=True, 
                             loss='CrossEntropy',
                             param_counts=np.array([3, 4, 7, 10, 15, 20, 23, 27, 31, 32, 33, 34, 36, 38, 40, 41, 42, 60, 100, 150, 300, 800]),
                             generate_parameters=False,
                             max_epochs=6000,
                             batch_size=128,
                             seed=0,
                             reuse_weights=False)

outs = model.double_descent()

save_obj(outs, 'mlp-dd-experiment-no-reuse')