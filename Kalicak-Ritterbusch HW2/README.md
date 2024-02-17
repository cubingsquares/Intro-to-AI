# Overview

#### **Code Created by:** Matthew Ritterbusch and Jack Kalicak

#### **Due:** 9 February 2024

The code was implementing utilizing Dr. Zabaras's lecture slides he posted to his website and the d2l book. Four problems are solved within the set, each of which with their own solution file. These files can be run in Google Colabs.

# Problem 1: Linear Regression Implementation from Scratch (Section 3.4)

The answers to the following questions were derived and can be seen found in RItterbusch-Kalicak-Problem1.ipynb

A. What would happen if we were to initialize the weights to zero. Would the algorithm still work? What if we initialized the parameters with variance 1000 rather than 0.01?

- See line 3 for the variance adjustment to sigma = 0. Initializing the weights to 0 does not seem to cause any issues at first glance, but the issue of vanishing/exploding gradients can come into play during backpropagation. It is best practice to never initialize weights to 0. Using variance=1000 causes an extremely large first loss value and the algorithm tends to have errors for w, b greater than 1. This large variance value can cause issues of exploding gradients as well as unstable training (the updates to the weights during each iteration can become extremely large, making the optimization process difficult to control).

B. Experiment using different learning rates to find out how quickly the loss function value drops. Can you reduce the error by increasing the number of epochs of training?

- With a smaller learning rate (.005 << .03), the initial loss value is higher and it does not reach zero without extending the number of epochs. Increasing the number of epochs does reduce the error. Increasing the learning rate (0.5 >> 0.03) seems to lead to faster convergence, but this does introduce the potential for overshooting. Choosing a proper learning rate is key to the model's success.
- C. Try implementing a different loss function, such as the absolute value loss ‘(y hat - d2l.reshape(y, y hat.shape)).abs().sum()‘.
  - (a) Check what happens for regular data.
    - The loss values do not seem to converge to zero - rather, they hover around ~10 even when increasing the epochs. Despite this, the resulting error is interestingly still on the scale of ~0.01 - 0.1.
  - (b) Check whether there is a difference in behavior if you actively perturb some entries, such as y5 = 10000, of y.
    - Perturbing an entry to a value of 10000 significantly magnifies the loss value to ~600 and causes it to oscillate. However, when the mean squared loss is used, the loss is magnified up to ~100000, so clearly the mean squared loss function is much more sensitive to outliers.
  - (c) Can you think of a cheap solution for combining the best aspects of squared loss and absolute value loss?
    - One way is to just use the square root of the mean squared loss to reduce its sensitivity to outliers. This works fairly well as tested in the above code. Another solution is to use the Huber loss - setting a proper delta parameter can achieve error minimization close to the same speed as the square root of the mean squared error. However, the square root method seems to be more reliable (no delta parameter guessing needed).

# Problem 2: Concise Implementation of Linear Regression (Section 3.5)

The answers to the following questions were derived and can be seen found in Ritterbusch-Kalicak-Problem2.ipynb

- A. Review the PyTorch documentation to see which loss functions are provided. In particular, replace the squared loss with Huber’s robust loss function.
  - Full loss function list and replacement with Huber's robust loss function can be viewed in Ritterbusch-Kalicak-Problem2.ipynb
- B. How does the solution change as you vary the amount of data generated? Plot the estimation error for wˆ − w and ˆb − b as a function of the amount of data. Hint: increase the amount of data logarithmically rather than linearly, i.e., 5, 10, 20, 50, ..., 10,000 rather than 1000, 2000, ..., 10,000.
The answers to the following questions were derived and can be seen found in RItterbusch-Kalicak-Problem1.ipynb
- As shown above, increasing the amount of data sharply decreases the error, akin to the shape of a y ~ 1/x function.

# Weight Decay (Section 3.7)

The answers to the following questions were derived and can be seen found in Ritterbusch-Kalicak-Problem3.ipynb
- A. Experiment with the value of the regularization parameter λ in the estimation problem. Plot training and validation accuracy as a function of λ. What do you observe?
Can you identify an optimal λ
The last figure in Ritterbusch-Kalicak-Problem3.ipynb shows that the validation and training error converge as λ approaches 100.
**The optimal λ for this routine is approximately 100 as shown in the figure in the second portion of the Ritterbusch-Kalicak-Problem3.ipynb code titled Running the graph for multiple lambdas.**

# Problem 4: Saving the parameters of a neural network and reloading the NN later on

The answers to the following questions were derived and can be seen found in Ritterbusch-Kalicak-Problem4.ipynb. **In order to run successfully shared_drive_path should be replaced with the path to where the user would like the parameter .pth file to be saved.**
- Check the documentation of PyTorch and provide an example (extending those in the textbook) where the model parameters (w and b) are saved to be used again to initialize the model at a later time. This is very important for large scale NN implementations.
Implement this for one of the examples in Chapter 3.
- Implementation of the example can be seen in Ritterbusch-Kalicak-Problem4.ipynd. The reasoning behind parameter saving is multifaceted. You can assist in model persisetence (saving computation time and expense), allow for rapid deployment, and allow for reproducibility of results, which could also be beneficial for experimentation and iteration of code design.
