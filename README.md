# Introduction
Implemented LeNet5 from scratch in PyTorch to recognize digits. This model has been trained and tested on the MNIST Dataset. The URL for the web app is [here](https://lenet-digit-recognition-advaitkisar.streamlit.app/).

# Details of the Project
## Libraries Used
The libraries used in this project are listed below:-
1) PyTorch - Used for implementing the network viz. LeNet5
2) MatPlotLib - Used for plotting inferences
3) Numpy - Used for numerical computations
4) PIL - Used for processing image files in Python
5) IO - Used for managing and processing input files
6) Streamlit - Used for deploying the project as a web application.

## Dataset
The dataset used for this project was the MNIST dataset of 70000 single-digit grayscale images, split into 60000 images for the training set and 10000 images for the test set. These two datasets were loaded using PyTorch and stored in the tensor form for processing in later steps. They were further split into batch sizes of 100 samples for training and testing.

## Visualizing Data
The data was visualized for different samples of both datasets using MatPlotLib.

## LeNet5 Model
Implemented a LeNet5 Model from scratch using a class object, with \__init\__ and forward functions. The architecture used is as follows:

| Layer | Description | Input | Output |
| --- | --- | --- | --- |
| Layer 1 | <ol><li>Convolutional Layer with kernel size 5 X 5 and padding set to 2</li><li>Tanh Activation Function</li><li>Average Pooling Layer.</li></ol> | 1 X 28 X 28 | 6 X 14 X 14 |
| Layer 2 | <ol><li>Convolutional Layer with kernel size 5 X 5</li><li>Tanh Activation Function</li><li>Average Pooling Layer.</li></ol> | 6 X 14 X 14 | 16 X 5 X 5 |
| Layer 3 | <ol><li>Convolutional Layer with kernel size 5 X 5</li><li>Tanh Activation Function</li></ol> | 16 X 5 X 5 | 120 X 1 X 1 |
| Layer 4 | <ol><li>Flatten the Convolutional Layer into a Fully Connected Layer</li><li>Fully Connected Layer with input size 120 and output size 84.</li><li>Tanh Activation Function</li></ol> | 120 X 1 X 1 | 84 |
| Layer 5 | Fully Connected Layer with input size 84 and output size 10 | 84 | 10 |
| Layer 6 | Softmax Layer for predicting the class | 10 | 1 |
