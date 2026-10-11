# Deep Neural Network from Scratch (NumPy)

This repository contains the complete implementation of a **Deep Neural Network from Scratch** using Python and NumPy, built as part of the *Deep Learning Specialization* (Course 1: Neural Networks and Deep Learning, Week 4 Assignment).

Instead of relying on high-level frameworks like TensorFlow or PyTorch, this project implements all core mathematical operations—such as parameter initialization, forward propagation, cost computation, backpropagation, and gradient descent updates—purely from first principles.

---

## 🚀 Project Overview & Architecture

The architecture supports an **$L$-layer Neural Network** structured as:
$$\text{[LINEAR} \rightarrow \text{RELU]} \times (L-1) \rightarrow \text{LINEAR} \rightarrow \text{SIGMOID}$$

---

## 🛠️ Implemented Functions (The 10 Exercises)

Below are the core modules and functions implemented in this project:

1. **`initialize_parameters(n_x, n_h, n_y)`**: Initializes weights and biases for a 2-layer neural network using random scaling (`0.01`) and zeros.
2. **`initialize_parameters_deep(layer_dims)`**: Generalizes parameter initialization for an arbitrary $L$-layer deep neural network.
3. **`linear_forward(A, W, b)`**: Computes the linear transformation step ($Z = W A + b$).
4. **`linear_activation_forward(A_prev, W, b, activation)`**: Combines the linear step with non-linear activation functions (`ReLU` or `Sigmoid`).
5. **`L_model_forward(X, parameters)`**: Executes full forward propagation across all $L$ layers while caching intermediate values for backpropagation.
6. **`compute_cost(AL, Y)`**: Computes the binary cross-entropy cost function.
7. **`linear_backward(dZ, cache)`**: Computes gradients with respect to the linear output ($dW, db, dA_{prev}$).
8. **`linear_activation_backward(dA, cache, activation)`**: Performs backward propagation through the linear-activation layer using `relu_backward` or `sigmoid_backward`.
9. **`L_model_backward(AL, Y, caches)`**: Executes full backward propagation across the entire network in reverse order.
10. **`update_parameters(params, grads, learning_rate)`**: Updates network parameters ($W^{[l]}, b^{[l]}$) via standard Gradient Descent.

---

## 📦 Tech Stack & Requirements

* **Python 3.x**
* **NumPy** (Scientific computing and matrix operations)
* **Matplotlib** (Data visualization)
* **H5Py** (Dataset handling)

---

