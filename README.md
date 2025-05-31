# micrograd-cpp 🧠

My version of a small **reversed automatic differentiation engine**, along with a multi-layer perceptron that can be trained and tested for accuracy. 

The project is made in _**Visual Studio 2022**_.



## Results

I managed to train this model to an **accuracy of 97.5%** on the [Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) using the following settings:

**Epochs:** 10

**Batch Size:** 8

**Learning Rate:** 0.001

**Momentum Coefficient:** 0.9

But I'm sure there are better configurations for an even better result.

## Features

### Implemented Activation Functions

* Sigmoid
* TanH
* ReLU
* LeakyReLU

### Implemented Loss Functions

* **Mean Squared Error**

### Other

* _**Mini-Batch Gradient Descent**_ optimization
* Implemented _**momentum functionality**_.
* Weight initialization is done with the _**He technique.**_
* _**80 / 20 Training / Testing**_ data split.
* **Data shuffling** each epoch.
* Ability to not apply activation function at the last layer. (Would be useful when applying functions such as softmax)
* Ability to train model over more sessions. (Saving weights and biases over time and loading them upon next execution)

### Things that can be improved / should be added

I encourage anyone interesed to add/fix any of these features to contribute!

* **Softmax + CrossEntropy loss function implementation.** (I have tried multiple times, but for some reason my implementation of the softmax function results in very small gradients everytime)
* Adding any more desired activation functions
* Creating a designated learning rate scheduler.
* Anything else you think this project might need! :)

### Credits

Inspired by karpathy's [micrograd](https://github.com/karpathy/micrograd).

## License

This project is under the  [MIT License](https://github.com/VladBonciu/micrograd-cpp/blob/main/LICENSE), so have fun with it!
