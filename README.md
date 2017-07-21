# NeuralNetworksInPython

Made by Chris Kormaris

Programming Language: Python


## NeuralNetworkForMNIST

Unzip the MNIST data from the file "mnisttxt.zip", in the same directory where the Python files are.
Then, run the Neural Network python code from the file "NN_mnist.py".

Neural Network details
File:
"NN_mnist.py"
1st Activation Function: sigmoid
2nd Activation Function: softmax
Maximum Likelihood Estimate Function: Cross Entropy Function
train algorithm: gradient ascent
Bias terms are used.

The precision that is achieved is: 94.7 %


## NeuralNetworksSpamHamClassification

Unzip the files "TRAIN.zip" and "TEST.zip", in the same directory where the Python files are.
First, run the python file "FeatureSelectionUsingIG.py" to generate the output file
"feature_dictionary.txt", containing the features tokens that we'll use.
Then run the Neural Network of your choice between "SpamHamNeuralNetworkSqErr.py" and "SpamHamNeuralNetworkCrossEntropy.py".

Note: You can use your own Train and Test text files if you want, as long as they contain "spam" or "ham" in their names, accordingly to their category. The existence of the substrings "spam" or "ham" in a text file define in which category of the two the text file belongs to.

### Neural Network #1
File:
"SpamHamNeuralNetworkSqErr.py"
1st Activation Function: tanh
2nd Activation Function: sigmoid
Loss Function: Mean Squared Error Loss

The first neural network that has been implemented uses
the tanh activation function in the first input layer
and the sigmoid activation function in the last output layer.
In the last node of the network the Squared Error Loss formula (SqErr loss function) is used.
The precision that is achieved is: 94.33%.

### Neural Network #2
File:
"SpamHamNeuralNetworkCrossEntropy.py"
1st Activation Function: sigmoid
2nd Activation Function: softmax
Loss Function: Cross Entropy Loss

The second neural network that has been implemented uses
the sigmoid activation function in the first input layer
and softmax sigmoid activation function in the last output layer.
In the last node of the network the Cross Entropy Loss formula is used.
The precision that is achieved is: 94.57%.
