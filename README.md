# Pattern-Recognition
1. The nearest neighbour classifier is built from scratch using the Euclidean distance measure and maps the neighbour with minimal distance (20 newsgroup tasets). This model is compared with the sklearn nearest neighbour classifier, and the sklearn dummy classifier is used as the baseline.
   
2. Perceptron gradient descent is trained and tested for simple logical functions and for the Multi layer perceptron(MLP) is also trained and tested.
   
3. MLP with 2 hidden layer and an output layer is trained which accpets RGB image as input(Vanilla Neural Network).
   Input layer - 3 × 64 × 64-dimensional is flattened
   1st hidden layer - 100 nodes
   2nd hidden layer - 100 nodes
   Output layer     - 2 nodes
   
4. CNN is implemented using PyTorch with German Traffic Sign Recognition Benchmark (GTSRB) dataset.(loss- cross entropy, optimizer - SGD)
   CNN architecture is given below,
   1st conv layer(2D) - 10 filter of size 3*3 with stide 2 and RELU acti. func.
   Max pool           - 2*2
   2nd conv layer     -  10 filter of size 3*3 with stide 2 and RELU acti. func.
   Max pool           - 2*2  (flattened before passing to next layer)
   fully connected Dense layer - 2 neurons with sigmoid acti. func.

5. RNN is configured for predicting next letter by utlising previous 2 letters
   (xt+1 = MLP(xt, xt−1)).
