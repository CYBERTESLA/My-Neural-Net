import numpy as np

# XOR Dataset
arr_x = np.array([[0,0], [0,1], [1,0], [1,1]])
arr_y = np.array([[1], [0], [0], [1]])

    

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

    # def sigmoid(self, x):
    #     # The sigmoid activation function
    #     return 1 / (1 + np.exp(-x))
    
    def sigmoid(self, x):
        out = np.zeros_like(x)
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        out[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        out[negative_mask] = exp_x / (1 + exp_x)
        return out

    
    # def dsigmoid_dx(self, x):
    #     # First derivative of the sigmoid activation function
    #     return x * (1-x)

    def dsigmoid_dx(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)


    def forward(self, X):
        # Forward Propagation through the network
        # Implementing wX+b

        # X is z1
        self.z2 = np.dot(X, self.weights_01) + self.b01 # operating from input layer to hidden layer (z2)
        self.a1 = self.sigmoid(self.z2) # applying activation function (a2)
        self.z3 = np.dot(self.a1, self.weights_12) + self.b12 # operating from hidden layer to output layer (z3)
        self.a2 = self.sigmoid(self.z3) # Final output (y_hat) == (a3)

        # print(self.weights_12)

        return self.a2.item()

        # print(self.a2)

    def Cost(self, X, y):
        # Function to find the Cost of our Model
        sum = 0

        losses = []

        for i in range(int((arr_x.size)/2)): # Runs for each training example
            loss = 0.5 * ((y[i] - self.forward(X[i])) ** 2) # Calculating Loss of each training example
            sum = sum + loss # Adding all losses to find cost

        losses.append(sum.item())
        print(losses)
            
        return sum.tolist(), losses
    
    def Cost_derivative(self, X, y): # X is the input dataset (arr_x) and y is the output dataset (arr_y)
        self.yHat = self.forward(X)

        delw3 = np.multiply(-(y - self.yHat), self.dsigmoid_dx(self.z3)) # Computing delta3, refer to equation 6 in notes.

        dJdW2 = np.dot(self.a2.T, delw3)

        delw2 = np.dot(delw3, self.weights_12.T) * self.dsigmoid_dx(self.z2)
        
        dJdW1 = np.dot(X.T, delw2)

        # delb3 = np.multiply(-(y - self.yHat), dsigmoid_dx(self.z3))

        # dJdb2 = delb3

        # dJdb1 = np.dot(self.a2.T, delb3) * dsigmoid_dx(self.z2)
        
        return dJdW2, dJdW1 #, dJdb2, dJdb1
    

    def train(self):
        # Using Gradient Descent here

        # dJdW2, dJdW1, dJdb2, dJdb1 = self.Cost_derivative(self.train_data, self.target_values)


        for i in range(self.epochs):

            dJdW2, dJdW1 = self.Cost_derivative(self.train_data, self.target_values)
            self.weights_12 -= self.lr * dJdW2
            self.weights_01 -= self.lr * dJdW1

            # print(self.lr)
            # print(self.b12)
            # print(dJdb2)
            # self.b12 -= np.array([[self.lr], [self.lr], [self.lr], [self.lr]]) * dJdb2
            # self.b01 -= self.lr * dJdb1

    def predict(self, X):
        
        npX = np.array(X)
        output = self.forward(npX)
        if (output >= 0.5):
            return 1
        else:
            return 0 

Network = NN(arr_x, arr_y)
Network.forward(np.array([1,1]))

# print(Network.forward(np.array([1,1])))

print(Network.Cost(arr_x, arr_y))

# Network.train()

# print(Network.Cost())

# print(Network.predict([1, 1]))
# print(Network.losses)







