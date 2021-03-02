import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np

class Plotter:
    """This class has tools for plotting the double descent curve for different models"""
    
    def __init__(self):
        pass
    
    def plot_random_forest(self, collected_data):
        """Plot double descent with the dictionary returned after training the scikit-learn
        Random Forest classifier. The plots are saved to the current directory
        
        ...
        Parameters
        ----------
        collected_data : dict
            The dictionary obtained by running double_descent on the RandomForest model
        """
        
        custom_ticks_label = [] 
        custom_ticks_x = []
        for i in range(len(collected_data['leaf_sizes'])):

            if i % 7 == 0:

                custom_ticks_label.append(
                    str(collected_data['leaf_sizes'][i]) + ' / ' + str(collected_data['trees'][i]))
                custom_ticks_x.append(i)
                
        plt.figure(figsize=(10, 20))
        fig, ax1 = plt.subplots()

        plt.title('Random Forest on MNIST')

        color = 'tab:blue'
        ax1.set_xlabel('Model parameters N_max_leaf/N_tree')
        ax1.set_ylabel('Squared Loss')
        ax1.plot(range(len(collected_data['mse_loss'])), collected_data['mse_loss'], color=color)
        ax1.set_ylim(0, max(collected_data['train_loss']))
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:orange'
        ax2.plot(range(len(collected_data['train_loss'])), collected_data['train_loss'], color=color)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_ylim(0, max(collected_data['train_loss']))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped


        train = mlines.Line2D([], [], color='tab:orange',
                                  markersize=15, label='Train')

        test = mlines.Line2D([], [], color='tab:blue',
                                  markersize=15, label='Test')

        plt.legend(handles=[train, test])

        plt.xticks(custom_ticks_x, custom_ticks_label)


        os.mkdir('random-forest-figures')
        plt.savefig('random-forest-figures/dd_random_forest_squared.jpg')
        plt.close()
        
        # ---------------------------------------------------
        plt.clf()
        plt.figure(figsize=(10, 20))
        
        fig, ax1 = plt.subplots()

        plt.title('Random Forest on MNIST')

        color = 'tab:blue'
        ax1.set_xlabel('Model parameters N_max_leaf/N_tree')
        ax1.set_ylabel('Zero-One Loss (%)')
        ax1.plot(range(len(collected_data['zero_one_loss'])), collected_data['zero_one_loss'], color=color)
        ax1.set_ylim(0, max(collected_data['train_loss']))
        
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.plot(range(len(collected_data['train_loss'])), collected_data['train_loss'], color=color)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_ylim(0, max(collected_data['train_loss']))
        fig.tight_layout()  

        train = mlines.Line2D([], [], color='tab:orange',
                                  markersize=15, label='Train')

        test = mlines.Line2D([], [], color='tab:blue',
                                  markersize=15, label='Test')


        plt.legend(handles=[train, test])

        plt.xticks(custom_ticks_x, custom_ticks_label)
        
        plt.savefig('random-forest-figures/dd_rf_zero_one.jpg')
        plt.close()
