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

## Training the Model
1) The LeNet5 was trained for 50 epochs using an Adam Optimizer with a learning rate of 0.001 and Cross Entropy Loss was used as the objective function.
2) In each epoch, the first for loop runs through the training dataset batch-wise and then, we find the accuracy of the model using the test set.
3) Both the training cost and the model accuracy on the test set were plotted against the number of epochs, and the final accuracy of the model after training was **98.56 %**.
4) The parameters of the model were saved in a '.pth' file in the directory, which can be fetched later to deploy the model without retraining.

## Prediction Section
In this section, the model was tested manually on the test set to check its performance.

## Testing Real-World Examples
1) After taking the input from the user, the image is center cropped to the minimum of height and width.
2) The image is then converted from RGB to Grayscale and the color is inverted so that the image is similar to the data on which the model has been trained for better detection.
3) Further, the image is resized to 28 X 28 pixels, converted to PyTorch Float 32 Tensor, and the dimension is expanded to (1, 1, 28, 28) to comply with the dimensions of the model parameters.
4) This modified input is passed through the model, which recognizes the digit and gives us the probability for the same.

## Deployment
1) I have used 'streamlit' for deploying the model using a web application, and the file for this implementation is 'digit_recognition.py'.
2) The model is loaded using the saved parameters in the file 'model.pth'.
3) The Streamlit website downloads and installs the necessary libraries in its virtual environment.
4) On the web app, the user has two options to give his/her input, either by uploading the file or by drawing a doodle.
5) The 1st option prompts the user to upload an image from his/her local system.
6) The 2nd option gives the user a canvas on which he/she can draw the digit by configuring the brush, and the canvas is sent to the Python script mentioned above for real-time processing.

# Inference
1) LeNet5 has been trained well enough to recognize single digits with an accuracy of **98.56 %**.
2) The model can be used by any user through the [web application](https://lenet-digit-recognition-advaitkisar.streamlit.app/) to make use of this to recognize single digits.
