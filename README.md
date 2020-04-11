# NeuralNetworksInPython

Made by Chris Kormaris

Programming Language: Python

The Neural Networks implemented here are CNNs (Convolutional Neural Networks) that classify given data to multiple categories. The learning method is supervised, because during training, labeled data are used.


Slightly based on code from this repository: <a href="https://github.com/dennybritz/nn-from-scratch">Implementing a Neural Network from Scratch in Python</a>

## Neural Network for Digit Classification of MNIST Dataset

Unzip the MNIST data from the compressed file "mnisttxt.zip" in the same directory where the Python files are.
Then, run the Neural Network that uses batch gradient ascent, from the file "NN_mnist.py" as follows:
```shell
python NN_mnist_batch_gradient_ascent.py
```
Alternatively, train the neural network using mini-batch gradient ascent. The batch size can be changed from within the code:
```shell
python NN_mnist_minibatch_gradient_ascent.py
```
Alternatively, train the neural network using stochastic gradient ascent. The batch size for stochastic gradient ascent is set to 1:
```shell
python NN_mnist_stochstic_gradient_ascent.py
```
Implementating the neural network using stochastic gradient ascent is very slow, because of the big amount of data of the MNIST dataset.

#### Neural Network details
File: "NN_mnist_batch_gradient_ascent.py"

1st Activation Function: tanh

2nd Activation Function: softmax

Maximum Likelihood Estimate Function: Cross Entropy Function

Train algorithm: Gradient Ascent

Bias terms are used.

The precision that was achieved, after training with batch gradient ascent, was: **94.67 %**

## Neural Networks for Spam-Ham Classification

Unzip the compressed file "LingspamDataset.zip" in the same directory where the Python files are.
First, run the python file "FeatureSelectionUsingIG.py" to generate the output file
"feature_dictionary.txt", containing the features tokens that we'll use.
```shell
python FeatureSelectionUsingIG.py
```
Then, to download the necessary stopwords from the "nltk" package, run:
```shell
python nltk_download.py
```
And then, to construct the train and test data, run:
```shell
python main.py
```
Then, run the Neural Network of your choice between **"NN_SpamHam_MSE_batch_gradient_descent.py"** and **"NN_SpamHam_CrossEntropy_batch_gradient_descent.py"**.
You can alternatively train the neural network using mini-batch gradient descent. The batch size can be changed from within the code. Run the files **"NN_SpamHam_MSE_minibatch_gradient_descent.py"** or **"NN_SpamHam_CrossEntropy_minibatch_gradient_descent.py"**.
You can alternatively train the neural network using stochastic gradient descent. Run the files **"NN_SpamHam_MSE_stochastic_gradient_descent.py"** or **"NN_SpamHam_CrossEntropy_stochastic_gradient_descent.py"**.

### Neural Network #1
Run:
```shell
python NN_SpamHam_MSE_batch_gradient_descent.py
```
File: "NN_SpamHam_MSE_batch_gradient_descent.py"

1st Activation Function: tanh

2nd Activation Function: sigmoid

Loss Function: Mean Squared Error Loss

Train algorithm: Gradient Descent


The first neural network that has been implemented uses
the tanh activation function in the first input layer
and the sigmoid activation function in the last output layer.
In the last node of the network the Mean Squared Error Loss formula (MSE loss function) is used.

The precision that was achieved, after training with batch gradient descent, was: **97.69 %**.

### Neural Network #2
Run:
```shell
python NN_SpamHam_CrossEntropy_batch_gradient_descent.py
```
File: "NN_SpamHam_CrossEntropy_batch_gradient_descent.py"

1st Activation Function: sigmoid

2nd Activation Function: softmax

Loss Function: Cross Entropy Loss

Train algorithm: Gradient Descent


The second neural network that has been implemented uses
the sigmoid activation function in the first input layer
and softmax sigmoid activation function in the last output layer.
In the last node of the network the Cross Entropy Loss formula is used.

The precision that was achieved, after training with batch gradient descent, was: **97.69 %**.
