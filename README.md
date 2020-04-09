# deep-neural-network-from-first-principles
Framework to build a highly customizable, fully connected deep neural network, using only numpy. 

The original purpose of this code was as a learning instrument, to show that I could create a deep neural network from first principles, using only numpy, without using any of the standard deep learning modules. I have also used it as a benchmarking tool, to compare speeds/accuracies against other deep learning modules such as tensorflow, keras, pytorch, etc.

For simplicity, the only class is the NeuralNetwork() class. Within it you'll find all the necessary functions/parameters that are required to customize a DNN. NeuralNetwork() is able to handle both binary as well as multi-label categorical classification.

The __name__ == '__main__' block near the end of the script demonstrates how to construct a multi-layered neural network, train it on a training set, and then test the trained network out on a test set. I tested it out on the MNIST dataset (m=60000) for digit recognition, and after only 250 iterations it was able to achieve ~64% accuracy.

<b>Loss functions:</b>
 - binary cross entropy
 - categorical cross entropy
 - <i>mean-squared error (not yet implemented)</i>
 - <i>hinge loss (not yet implemented)</i>
 
 <b>Parameter Initialization:</b>
 - Zeros
 - Ones
 - Random
 - Xavier
 
 <b>Metrics:</b>
 - Accuracy
 - <i>Precision (not yet implemented)</i>
 - <i>Recall (not yet implemented)</i>
 - <i>F1 score (not yet implemented)</i>
 
 <b>Future improvements:</b>
 - Implement options for training optimization algorithms (Adam, Adagrad, etc)
 - Mini-batching customization
 - 2D and 3D Convolutional layers
 - 2D and 3D pooling layers
