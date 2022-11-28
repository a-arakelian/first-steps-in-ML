> MISSING FILES:
To work properly, you need to download the [following 7z file](https://www.dropbox.com/s/tjglzu9fzb6egrz/Dataset.7z?dl=0) from the dropbox and extract the files in this directory.

### Table of Contents:

-  [MLP manually](#mlp-manually)

## MLP manually
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
    <br />
    <a href="https://academy.yandex.ru/handbook/ml/article/metod-obratnogo-rasprostraneniya-oshibki"><img src="https://yastatic.net/s3/ml-handbook/admin/17_4_b1b2356957.gif"></a>
    <br />
</p>
