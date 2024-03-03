# Overview - HW3

#### **Code Created by:** Matthew Ritterbusch and Jack Kalicak

#### **Due:** 17 February 2024

**Note:** If `d2l` is not locally installed, you must first install it using `!pip install d2l==1.0.3`. Currently, the programs interface with `d2l` stored on Google Drive. These first lines can be commented out and replaced with the `d2l` pip install.

## P1. Concise Implementation of Softmax Regression (Section 4.5)

### Experiment with the hyperparameters of the code in this section.

**A. Plot how the validation loss changes as you change the learning rate.**

- The smaller the learning rate, the larger the loss is initially, as the model takes smaller steps during training.
- Problem 4 will evaluate how increasing the number of epochs may allow these learning rates to ultimately converge.

**B. Do the validation and training loss change as you change the minibatch size?**

- Both losses start higher with the larger batch size but appear to approach convergence as the epochs increase.
- The larger batch size promotes a steadier descent and fewer fluctuations in the validation loss.
- Smaller minibatches exhibit more variability in gradient estimates, leading to noisier estimates.
- Power-of-two differences from the standard 256 minibatch size often reveal differences in loss performance.

## P2. Softmax Regression Implementation from Scratch (Section 4.4)

**A. Test whether softmax still works correctly if an input has a value of 100.**

- Extremely high input values (like 100) cause exploding issues due to the exponential, leading to an infinite sum of the softmax output tensor.

**B. Test whether softmax still works correctly if the largest of all inputs is smaller than -100?**

- Extremely small input values cause underflow issues due to the exponential.

**C. Implement a fix by looking at the value relative to the largest entry in the argument.**

- Subtracting the largest value of the input matrix from all elements ensures the softmax function works properly and the output elements sum to 1.

## P3. Softmax Regression Implementation from Scratch (Section 4.4)

- The new cross-entropy loss function takes much longer to run due to elementwise matrix multiplication, which is computationally expensive.
- To prevent issues with logarithms being undefined for zero or negative values, clip predicted probabilities with a small positive value or add a small constant before taking the logarithm.

## P4. Softmax Regression Implementation from Scratch (Section 4.4)

**A. Increase the number of epochs for training. Why might the validation accuracy decrease after a while? How could we fix this?**

- Overfitting and learning rate decay can cause validation accuracy to decrease over time.
- Strategies to mitigate this include using dropout for regularization, early stopping, and modifying the learning rate during training.

**B. What happens as you increase the learning rate? Compare the loss curves for several learning rates. Which one works better? When?**
When you increase the learning rate the losses are intially higher, but converge over time. However the 0.001 rate looks like it overfit intially. Therefore 0.01 seemed to be the best model of the data. For small number of epochs a higher learning rate enables quicker, but less accurate results. Finding this balance with 0.01 was key.  
