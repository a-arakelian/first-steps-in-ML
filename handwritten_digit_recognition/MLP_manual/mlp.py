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
            
### another try ###
def softmax_differential(x: np.array):
    differential = []
    (n, m) = x.shape
    for i in range(n):
        v = x[i,:].reshape((1, m))
        _differential = np.diag(v.reshape(-1)) - (v.T @ v)
        differential.append(_differential)
    return(np.array(differential))

ACTIVATION_FUNCTIONS = {
    'id': [lambda x: x, lambda x: 1, False],
    'tanh': [lambda x: (np.exp(2*x) - 1)/(np.exp(2*x) + 1), lambda x: 1 - (np.exp(2*x) - 1)/(np.exp(2*x) + 1)**2, False],
    'sigmoid': [lambda x: 1/(np.exp(-x) + 1), lambda x: (1/(np.exp(-x) + 1))*(1/(np.exp(x) + 1)), False],
    'softmax': [lambda x: np.exp(x)/(np.exp(x).sum(axis=1).reshape(x.shape[0], 1)), softmax_differential, True]
}
LOSS_FUNCTIONS = {
    'mse': [lambda x, y: ((x - y)*(x - y)).sum()/(y.shape[0]), lambda x, y: (1/y.shape[0])*2*(x - y)],
    'negative_log_likelihood': [lambda x, y: -(y * np.log(x)).sum(), lambda x, y: (y * np.log(x)).sum(),],
    'softmax_negative_log_likelihood': [lambda x, y: -(1/y.shape[0])*(y * np.log(ACTIVATION_FUNCTIONS['softmax'][0](x))).sum(), lambda x, y: (1/y.shape[0])*(ACTIVATION_FUNCTIONS['softmax'][0](x) - y)]
}


class Layer:
    def __init__(self, input_size = 5, output_size = 2, activation_function_name = 'tanh', children = None, parent = None):
        self.tensor = ACTIVATION_FUNCTIONS[activation_function_name][2]
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size) # +1 is bias
        self.input_with_bias = self._input_data_with_bias(np.zeros((1, input_size)))
        self.liner_transform = np.zeros((1, output_size))
        self.liner_transform.shape = (1, output_size)
        self.gradient = np.zeros((input_size + 1, output_size))
        self.gradient.shape = (input_size + 1, output_size)
        self.children = children
        self.parent = parent
        self.activat = ACTIVATION_FUNCTIONS[activation_function_name][0]
        self.gradient_of_activat = ACTIVATION_FUNCTIONS[activation_function_name][1]
        
    def forward_pass(self, input_data: np.array):
        self.input_with_bias = self._input_data_with_bias(input_data)
        self.liner_transform = self.input_with_bias @ self.weights
        activation = self.activat(self.liner_transform)
        if self.children != None:
            return self.children.forward_pass(activation)
        else:
            return activation
    
    def backward_pass(self, gradient):
        eye = np.eye(self.output_size, self.output_size+1, 1) if self.children else np.eye(self.output_size, self.output_size)
        #print('gradient size does not match to output size') if gradient.shape[1] != self.output_size else None
        activation_gradient = self.gradient_of_activat(self.liner_transform)
        liner_transform_gradient = self.input_with_bias
        if self.tensor:
            # activation_gradient has shape (batchsize, n, n)
            np.matmul(np.expand_dims((gradient @ eye.T), 1), activation_gradient).squeeze()
        else:
            self.gradient = liner_transform_gradient.T @ ((gradient @ eye.T) * activation_gradient)
        if self.parent != None:
            self.parent.backward_pass((gradient @ eye.T * activation_gradient) @ self.weights.T)
            
    def _input_data_with_bias(self, input_data):
        print('input_data does not match to input size') if input_data.shape[1] != self.input_size else None
        input_data.shape = (-1, self.input_size)
        input_with_bias = np.ones((input_data.shape[0], self.input_size+1))
        input_with_bias.shape = (input_data.shape[0], self.input_size+1)
        input_with_bias[:,1:] = input_data
        return input_with_bias


class MLP2:
    def __init__(self, layers = [(784, ), (196, 'tanh'), (49, 'tanh'), (10, 'sigmoid')], loss_function_name = 'mse'):
        self.layers = [
            Layer(layers[i][0], layers[i+1][0], layers[i+1][1])
            for i in range(len(layers)-1)
        ]
        for i in range(len(self.layers)-1):
            self.layers[i].children = self.layers[i+1]
            self.layers[i+1].parent = self.layers[i]
        self.loss_func = LOSS_FUNCTIONS[loss_function_name][0]
        self.gradient_of_loss_func = LOSS_FUNCTIONS[loss_function_name][1]
    def forward_propagat(self, input_data: np.array):
        return self.layers[0].forward_pass(input_data) 
        
    def backward_propagat(self, target: np.array, pred: np.array):
        gradient_of_loss = self.gradient_of_loss_func(pred, target)
        self.layers[-1].backward_pass(gradient_of_loss)
        
    def fit(self, X_train: np.array, Y_train: np.array, ephocs = 10, learning_rate = 2, show_los = True, X_test = False, Y_test = False, batch_size = 100, shuffle = True):
        if batch_size == None:
            batch_size = len(X_train)
        baths = self._get_baths(X_train, Y_train,batch_size, shuffle)
        error_history_train = []
        error_history_test = []
        for _ in range(int(ephocs/len(baths)) + 1):
            for (X, Y) in baths:
                pred = self.forward_propagat(X)
                self.backward_propagat(Y, pred)
                self._change_weights(learning_rate)
                if show_los:
                    error_history_train.append(self.loss_func(self.forward_propagat(X_train), Y_train))
                    if type(X_test) != bool  and type(Y_test) != bool:
                        error_history_test.append(self.loss_func(self.forward_propagat(X_test), Y_test))
        if show_los:
            plt.plot(error_history_train)
            plt.show()
            if type(X_test) != bool  and type(Y_test) != bool:
                plt.plot(error_history_test)
                plt.show()
    def _get_baths(self, X, Y, batch_size, shuffle = True):
        print('len of X fo not match to len of Y') if len(X) != len(Y) else None
        if shuffle:
            self._shuffle(X, Y)
        number_of_baths = len(X) // batch_size
        baths = []
        for i in range(number_of_baths):
            if i != number_of_baths-1:
                baths.append((X[i*batch_size: (i+1)*batch_size], Y[i*batch_size: (i+1)*batch_size]))
            elif i * number_of_baths != len(X):
                baths.append((X[i*batch_size:], Y[i*batch_size:]))
        return baths
        
    def _shuffle(self, X: np.array, Y: np.array):
        print('len of X fo not match to len of Y') if len(X) != len(Y) else None
        indexes = np.arange(0, len(X), 1)
        np.random.shuffle(indexes)
        X = X[indexes]
        Y = Y[indexes]
        
    def _change_weights(self, learning_rate):
        for i in range(len(self.layers)):
            self.layers[i].weights -= learning_rate * self.layers[i].gradient
