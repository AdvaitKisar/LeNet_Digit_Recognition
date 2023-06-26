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
