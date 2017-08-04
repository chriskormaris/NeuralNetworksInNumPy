# NeuralNetworksInPython

Made by Chris Kormaris

Programming Language: Python



## NeuralNetworkForMNIST

Unzip the MNIST data from the file "mnisttxt.zip", in the same directory where the Python files are.
Then, run the Neural Network python code from the file "NN_mnist.py" as follows:
```python
python NN_mnist.py
```

Neural Network details
File: "NN_mnist.py"

1st Activation Function: sigmoid

2nd Activation Function: softmax

Maximum Likelihood Estimate Function: Cross Entropy Function

Train algorithm: Gradient Ascent

Bias terms are used.

The precision that was achieved: 94.7 %


## NeuralNetworksSpamHamClassification

Unzip the files "TRAIN.zip" and "TEST.zip", in the same directory where the Python files are.
First, run the python file "FeatureSelectionUsingIG.py" to generate the output file
"feature_dictionary.txt", containing the features tokens that we'll use.
```python
python FeatureSelectionUsingIG.py
```
Then, run the Neural Network of your choice between "SpamHamNeuralNetworkSqErr.py" and "SpamHamNeuralNetworkCrossEntropy.py".

**Notes:**
<ol>
<li>You can use your own Train and Test text files if you want, as long as they contain "spam" or "ham" in their names, according to their category. The existence of the substring "spam" or "ham" in a text file defines in which category of the two the text file belongs to.</li>
<li>For the purpose of reserving the minimum amount of memory, I have converted the classification parameter arrays X and y to the appropriate data type, which is array of integers.
</li>
</ol>

### Neural Network #1
Run:
```python
python SpamHamNeuralNetworkSqErr.py
```
File: "SpamHamNeuralNetworkSqErr.py"

1st Activation Function: tanh

2nd Activation Function: sigmoid

Loss Function: Mean Squared Error Loss

Train algorithm: Gradient Descent


The first neural network that has been implemented uses
the tanh activation function in the first input layer
and the sigmoid activation function in the last output layer.
In the last node of the network the Squared Error Loss formula (SqErr loss function) is used.
The precision that was achieved: 94.33%.

### Neural Network #2
Run:
```python
python SpamHamNeuralNetworkCrossEntropy.py
```
File: "SpamHamNeuralNetworkCrossEntropy.py"

1st Activation Function: sigmoid

2nd Activation Function: softmax

Loss Function: Cross Entropy Loss

Train algorithm: Gradient Descent


The second neural network that has been implemented uses
the sigmoid activation function in the first input layer
and softmax sigmoid activation function in the last output layer.
In the last node of the network the Cross Entropy Loss formula is used.
The precision that was achieved: 93.75%.
