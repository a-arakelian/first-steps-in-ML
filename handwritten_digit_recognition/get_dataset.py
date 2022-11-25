import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

hdr_directory = os.path.abspath('')

class DigitExample:
    def __init__(self, digits = ['3'], n = 1):
        self.digits = digits
        self.folders = [
            (hdr_directory + '/' + 'Dataset' + '/' + 'testing' + '/' + digit, digit)
            for digit in digits
        ]
        self.digits_file = {}
        self._update_digits(n)
        
    def _update_digits(self, n = 1):
        for path in self.folders:
            arr = os.listdir(path[0])
            self.digits_file[path[1]] = [path[0] + '/' + i for i in random.sample(arr, n)]

    def matplot_show(self, update = False, n = 1):
        self._update_digits(n) if (update or n) else None
        if n * len(self.digits) > 1:
            fig, axs = plt.subplots(
                len(self.digits), n,
                figsize=(1.2*n, 1.2*len(self.digits)),
                layout="constrained"
            )
            for i in range(len(self.digits)):
                for j in range(n):
                    digit = self.digits[i]
                    path = self.digits_file[digit][j]
                    axs.flat[i*n + j].imshow(mpimg.imread(path))
                    axs.flat[i*n + j].set_title(digit)
            plt.show()
        else:
            digit = self.digits[0]
            path = self.digits_file[digit][0]
            plt.imshow(mpimg.imread(path))
        
    def numpy_show(self, update = False, n = 1):
        self._update_digits(n) if (update or n) else None
        for i in range(len(self.digits)):
            for j in range(n):
                digit = self.digits[i]
                path = self.digits_file[digit][j]
                print('The digit is ' + digit)
                print(mpimg.imread(path))

def digit2array(digit):
    digit_array = np.zeros(10)
    digit_array[digit] = 1
    return digit_array

def array2digit(array):
    return array.argmax()

def show_array_as_image(array):
    plt.imshow(array)

def from_dataset_jpgs2array(training_data = True, N = None):
    folder_name = 'training' if training_data else 'testing'
    folders = [
        (hdr_directory + '/' + 'Dataset' + '/' + folder_name + '/' + str(digit), digit)
        for digit in range(10)
    ]
    digit_file_names = {}
    digit_file_names_len = []
    for path in folders:
        arr = os.listdir(path[0])
        digit_file_names[path[1]] = [path[0] + '/' + i  for i in arr if i.endswith(".jpg")]
        digit_file_names_len.append(len(digit_file_names[path[1]]))
    if N == None: N = min(digit_file_names_len)
    digit_arrays = {}
    for i in range(10):
        digit_arrays[i] = [mpimg.imread(path) for path in digit_file_names[i][:N]]
    return digit_arrays
