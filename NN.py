import numpy as np

# XOR Dataset
arr_x = np.array([[0,0], [0,1], [1,0], [1,1]])
arr_y = np.array([[1], [0], [0], [1]])

arr_w_01 = np.array([])
# def forward():
    
def sigmoid(x):
    # The sigmoid activation function
    return 1 / (1 + np.exp(-x))
    
def dsigmoid_dx(x):
    # First derivative of the sigmoid activation function
    return x * (1-x)
class NN():

    # Specify Hyperparameters i.e. The Architecture of the Network.
    def __init__(self, train_data, target, learning_rate=0.1, epochs=100, num_inputs=2, num_hidden=2, num_output=1):
        self.train_data = train_data
        self.target_values = target
        self.lr = learning_rate
        self.epochs = epochs


        # initializing weights randomly
        self.weights_01 = np.random.uniform(size=(num_inputs, num_hidden)) # weights between input and hidden layer (w1)
        self.weights_12 = np.random.uniform(size=(num_hidden, num_output)) # weights between hidden and output layer (w2)

        # initializing biases randomly
        self.b01 = np.random.uniform(size=(1,num_hidden)) # biases for hidden layer (w1)
        self.b12 = np.random.uniform(size=(1,num_output)) # biases for output layer (w2)


    def forward(self, X):
        # Forward Propagation through the network
        # Implementing wX+b

        # X is z1
        self.z2 = np.dot(X, self.weights_01) + self.b01 # operating from input layer to hidden layer (z2)
        self.a1 = sigmoid(self.z2) # applying activation function (a2)
        self.z3 = np.dot(self.a1, self.weights_12) + self.b12 # operating from hidden layer to output layer (z3)
        self.a2 = sigmoid(self.z3) # Final output (y_hat) == (a3)

        return self.a2

        # print(self.a2)

    def Cost(self):
        # Function to find the Cost of our Model
        sum = 0
        for i in range(int((arr_x.size)/2)): # Runs for each training example
            loss = 0.5 * (arr_y[i] - self.forward(arr_x[i])) # Calculating Loss of each training example
            sum = sum + loss # Adding all losses to find cost
            
        return sum
    
    def Cost_derivative(self, X, y): # X is the input dataset (arr_x) and y is the output dataset (arr_y)
        self.yHat = self.forward(X)

        del3 = np.multiply(-(y - self.yHat), dsigmoid_dx(self.z3)) # Computing delta3, refer to equation 6 in notes.

        dJdW2 = np.dot(self.a2.T, del3)

        del2 = np.dot(del3, self.weights_12.T) * dsigmoid_dx(self.z2)
        
        dJdW1 = np.dot(X.T, del2)

        return dJdW2, dJdW1

        


Network = NN(arr_x, arr_y)
# Network.forward(np.array([1,1]))

print(Network.forward(np.array([1,1])))

print(Network.Cost())


# NEXT:Implement the Backpropagation Algorithm using Gradient Descent


