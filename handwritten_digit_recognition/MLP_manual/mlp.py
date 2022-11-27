import numpy as np
import matplotlib.pyplot as plt
    
class MLP:
    def __init__(self, input_size = 28*28, hidden_layers_sizes = [16*10, 8*10, 2*10], output_size = 10):
        # initialized attributes
        self.output_size = output_size
        self.sigma_activation =  lambda x: 1/ (np.exp(-x) + 1)
        self.sigma_differential = lambda x: self.sigma_activation(x)*self.sigma_activation(-x)
        self.layers_sizes = [input_size] + hidden_layers_sizes + [output_size]
        #Affine transformations between layers are performed by numpy matrices and arrays
        self.weights = []
        self.biases = []
        for i in range(len(self.layers_sizes) - 1):
            self.weights.append(np.random.rand(self.layers_sizes[i+1], self.layers_sizes[i]))
            self.biases.append(np.random.rand(self.layers_sizes[i+1], 1))
        # 'NOT' initialized attributes
        self.activations = []
        self.intensities = []
        self.gradients_of_weights = []
        self.gradients_of_biases = []
    def forward_propagat(self, data_in: np.array):
        data = self._prepare_input(data_in)
        self.activations = [data]
        self.intensities = [data]
        for i in range(len(self.weights)):
            affine_transform = self.weights[i]@self.activations[i] + self.biases[i]
            self.intensities.append(affine_transform)
            self.activations.append(self.sigma_activation(affine_transform))
        return self.activations[-1]
        
    def _prepare_input(self, data_in: np.array):
        data_in.shape = (self.layers_sizes[0], -1)
        return data_in
    
    def _prepare_output(self, target: np.array):
        target.shape = (self.output_size, -1)
        if target.shape[1] != self.activations[-1].shape[1]:
            print("Number of samples of data and target do not match!!!")
        return target
    
    def backward_propagat(self, target):
        y = self._prepare_output(target)
        # Loss function is sum_{by samples}(<y - y', y - y'>) / N_samples where y, y' \in R^d (d = 10)
        # As tr(A.T@B) = sum_ij(a_ij*b_ij) loss function is tr((Y_pr - Y).T@(Y_pr - Y)). So L(Y_pr) = tr((Y_pr - Y).T@(Y_pr - Y))
        # Gradient of L is 2(Y_pr - Y) in terms of [DL](H) = tr(2(Y_pr - Y)H.T)
        gradient_of_loss_function = (1/self.activations[-1].shape[1])*2*(self.activations[-1] - y)
        self.gradients_of_weights = []
        self.gradients_of_biases = []
        for i in reversed(range(len(self.weights))):
            #sigmoid_function(affine_transform())
            differential_of_sigmoid_function = self.sigma_differential(self.intensities[i+1])
            differential_of_affine_transform = self.activations[i]
            casteel = gradient_of_loss_function * differential_of_sigmoid_function
            gradient_of_weight = (gradient_of_loss_function * differential_of_sigmoid_function) @ differential_of_affine_transform.T
            array_of_ones = np.ones((1, self.activations[i].shape[1]))
            gradient_of_biase = (gradient_of_loss_function * differential_of_sigmoid_function) @ array_of_ones.T
            self.gradients_of_weights.append(gradient_of_weight)
            self.gradients_of_biases.append(gradient_of_biase)
            gradient_of_loss_function = self.weights[i].T @ casteel
            
    def fit(self, X_input, Y_target, number_of_epochs = 500, learning_rate = 2, show = False):
        error = []
        for k in range(number_of_epochs):
            Y_predict = self.forward_propagat(X_input)
            if show:
                error.append(((Y_predict - Y_target)*(Y_predict - Y_target)).sum()/(Y_predict.shape[1]))
            self.backward_propagat(Y_target)
            for i, j in enumerate(reversed(range(len(self.weights)))):
                self.weights[i] -= learning_rate * self.gradients_of_weights[j]
                self.biases[i] -= learning_rate * self.gradients_of_biases[j]
        if show:
            plt.plot(error)
            plt.show()

