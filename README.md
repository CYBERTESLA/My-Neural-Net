# My Neural Net

A simple neural network implementation in Python using NumPy to solve the XOR problem. This project demonstrates the fundamentals of forward propagation, backpropagation, and training using gradient descent.

## 🔍 Overview

This project implements a feedforward neural network with one hidden layer to learn the XOR logic gate. It includes:

- Manual implementation of forward and backward propagation
- Training using gradient descent
- Visualization of the loss curve over epochs using Matplotlib

## 📁 Project Structure

```
My-Neural-Net/
├── NN.py                # Main neural network implementation
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Running the Project

To train the neural network and visualize the loss curve:

```bash
python NN.py
```

## 📊 Visualizing Training Loss

The training process records the loss at each epoch. After training, a plot of the loss over epochs is displayed using Matplotlib, providing insight into the network's learning progress.

## 🧠 Neural Network Architecture

- **Input Layer**: 2 neurons (for the two inputs of the XOR problem)
- **Hidden Layer**: 2 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

## 🛠️ Features

- Custom implementation of sigmoid activation function and its derivative
- Forward propagation to compute predictions
- Backpropagation to compute gradients
- Weight and bias updates using gradient descent
- Loss tracking and visualization

## 📈 Example Output

Upon running the script, you should see the loss decreasing over epochs, indicating that the network is learning the XOR function.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact the repository owner.
