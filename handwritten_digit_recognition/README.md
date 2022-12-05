> MISSING FILES:
To work properly, you need to download the [following 7z file](https://www.dropbox.com/s/tjglzu9fzb6egrz/Dataset.7z?dl=0) from the dropbox and extract the files in this directory.

### Table of Contents:

-  [MLP manually](#mlp-manually)

## MLP manually
### first variant (MLP)
When implementing the MLP manually, as one might expect, the most tricky part was the backward propagation of the error. As the most complex and important part of the code, I will highlight it here.
```py
    def backward_propagat(self, target):
        y = self._prepare_output(target)
        gradient_of_loss_function = (1/self.activations[-1].shape[1])*2*(self.activations[-1] - y)
        self.gradients_of_weights = []
        self.gradients_of_biases = []
        for i in reversed(range(len(self.weights))):
            differential_of_sigmoid_function = self.sigma_differential(self.intensities[i+1])
            differential_of_affine_transform = self.activations[i]
            casteel = gradient_of_loss_function * differential_of_sigmoid_function
            gradient_of_weight = (gradient_of_loss_function * differential_of_sigmoid_function) @ differential_of_affine_transform.T
            array_of_ones = np.ones((1, self.activations[i].shape[1]))
            gradient_of_biase = (gradient_of_loss_function * differential_of_sigmoid_function) @ array_of_ones.T
            self.gradients_of_weights.append(gradient_of_weight)
            self.gradients_of_biases.append(gradient_of_biase)
            gradient_of_loss_function = self.weights[i].T @ casteel
```
Full MLP code can be viewed in the [`MLP_manual/mlp.py`](https://github.com/a-arakelian/first-steps-in-ML/blob/main/handwritten_digit_recognition/MLP_manual/mlp.py).

For a full understanding of the implementation of the method of error back propagation, it is necessary to understand well and get used to the notation of matrix differentiation.

<p align="center">
    <a href="https://academy.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki"><img src="https://yastatic.net/s3/ml-handbook/admin/17_4_b1b2356957.gif"></a>
</p>

#### Outcome

Pretty weird shit is going on. When performing gradient descent, the values of the `loss function` decrease, but the neural network from the word does *not* cope with the recognition of handwritten numbers at all, although it has pretty well learned multiplication. All results can be viewed in [`mlp.ipynd`](https://github.com/a-arakelian/first-steps-in-ML/blob/main/handwritten_digit_recognition/mlp.ipynb).
> [Later]([https://github.com/a-arakelian/first-steps-in-ML/edit/main/handwritten_digit_recognition/README.md#outcome-2) it became clear that the problem was in the wrong scaling of the data.

### second variant (MLP2)
>Given the oddities and slowness of the previous model, I decided to make another option. Changed the construction architecture by dividing one large class into classes `Layer` and `MLP2` and also added the ability to monitor the loss function on the learning sample and on the test

It seems that on the second attempt, the backward_pass turned out to be more compact. As usual, I'm posting it here.
```py
    def backward_pass(self, gradient):
        eye = np.eye(self.output_size, self.output_size+1, 1) if self.children else np.eye(self.output_size, self.output_size)
        activation_gradient = self.gradient_of_activat(self.liner_transform)
        liner_transform_gradient = self.input_with_bias
        if self.tensor:
            np.matmul(np.expand_dims((gradient @ eye.T), 1), activation_gradient).squeeze()
        else:
            self.gradient = liner_transform_gradient.T @ ((gradient @ eye.T) * activation_gradient)
        if self.parent != None:
            self.parent.backward_pass((gradient @ eye.T * activation_gradient) @ self.weights.T)
```
#### Outcome 2
After many unsuccessful attempts, I finally realized that the problem was not in the `MLP` at all, but only the output data was scaled unsuccessfully (from 0 to 1). By scaling the data on the interval 10 times smaller, that is, from 0 to 0.1, everything worked!
#### P.S.
In these attempts, I tried the `softmax` activation function and as a loss function `negative log likelihood`. But in order to avoid tensor multiplications, I took the `identical` mapping as an activation function, and loss took the `composition of softmax and negative log likelihood`. In this form, everything also worked, **which I can’t say about other implementation options, since I haven’t tested them**
