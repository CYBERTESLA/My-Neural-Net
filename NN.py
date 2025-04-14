import numpy as np

# XOR Dataset
arr_x = np.array([[0,0], [0,1], [1,0], [1,1]])
arr_y = np.array([[1], [0], [0], [1]])

arr_w_01 = np.array([])
# def forward():
    
def sigmoid(self, x):
    # The sigmoid activation function
    return 1 / (1 + np.exp(-x))
    
def dsigmoid_dx(self, x):
    # First derivative of the sigmoid activation function
    return x * (1-x)
class NN():

    # Specify Hyperparameters i.e. The Architecture of the Network.
    def __init__(train_data, target, learning_rate=0.1, epochs=100, num_inputs=2, num_hidden=2, num_output=1):
        self.train_data = train_data
        self.target_values = target
        self.lr = learning_rate
        self.epochs = epochs


        # initializing weights randomly
        self.weights_01 = np.random.uniform(size=(num_inputs, num_hidden)) # weights between input and hidden layer
        self.weights_12 = np.random.uniform(size=(num_hidden, num_output)) # weights between hidden and output layer

        # initializing biases randomly
        self.b01 = np.random.uniform(size=(1,num_hidden)) # biases for hidden layer 
        self.b12 = np.random.uniform(size=(1,num_output)) # biases for output layer


    def forward(self, X):
        # Forward Propagation through the network
        # Implementing wX+b
        self.h1 = np.dot(X, self.weights_01) + self.b01 # operating from input layer to hidden layer
        a1 = sigmoid(self.h1) # applying activation function
        self.op = np.dot(self.h1, self.weights_12) + self.b12 # operating from hidden layer to output layer
        a2 = sigmoid(self.op) # Final output



    


        
