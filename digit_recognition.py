from streamlit_drawable_canvas import st_canvas

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
    self.act1 = nn.Tanh()
    self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
    self.act2 = nn.Tanh()
    self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
    self.act3 = nn.Tanh()

    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(120, 84)
    self.act4 = nn.Tanh()
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    # Input - 1X28X28, Output - 6X28X28
    x = self.act1(self.conv1(x))
    # Input - 6X28X28, Output - 6X14X14
    x = self.pool1(x)
    # Input - 6X14X14, Output - 16X10X10
    x = self.act2(self.conv2(x))
    # Input - 16X10X10, Output - 16X5X5
    x = self.pool2(x)
    # Input - 16X5X5, Output - 120X1X1
    x = self.act3(self.conv3(x))
    # Input - 120X1X1, Output - 84
    x = self.act4(self.fc1(self.flat(x)))
    # Input - 84, Output - 10
    x = self.fc2(x)
    return x

def import_and_predict(img, model, p_threshold):
  img_transform = transforms.Compose([transforms.Grayscale(), transforms.RandomInvert(p=1)])
  img_new = img_transform(img)
  img_new.show()
  composed = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
  img_t = composed(img_new)
  img_t = img_t.type(torch.float32)
  x = img_t.expand(1, 1, 28, 28)
  z = model(x)
  z = nn.Softmax(dim=1)(z)
  p_max, yhat = torch.max(z.data, 1)
  p = float(format(p_max.numpy()[0], '.4f'))*100
  yhat_val = int(float(yhat.numpy()[0]))

  # Check if the image is blank (thresholding based on pixel values)
  img_array = np.array(img_new.convert("L"))  # Convert to grayscale
  pix_threshold = 252
  max_pixel = 255
  mean = np.mean(img_array)
  is_blank_image = mean > pix_threshold or mean < max_pixel-pix_threshold

  if p >=p_threshold and not is_blank_image:
      string = f"The uploaded image is of the digit {yhat_val} with {p:.2f} % probability."
      st.success(string, icon="✅")
  else:
      st.warning(f"The prediction probability is less than {p_threshold}% or the image is blank.", icon="⚠️")


def common_message():
  st.write('''
  ### Contact:
  For any queries or feedback, reach out to:
  - **Advait Amit Kisar**
  - Phone: +91 7774035501
  - Email: [advaitkisar2509@gmail.com](mailto:advaitkisar2509@gmail.com)
  
  Thank you for using this web app!
  ''')


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = LeNet5()
  model.load_state_dict(torch.load('./model.pth'))
  return model

# Add the sidebar header and project details
st.sidebar.header("Single Digit Recognition")
st.sidebar.write("""
This application leverages the LeNet architecture to recognize handwritten digits. Users can either upload an image of a digit or draw one directly on the canvas. The model predicts the digit along with the associated probability.

### Features:
- **Upload Image**: Upload a digit image in JPG or PNG format.
- **Draw a Doodle**: Use the drawing canvas to create a digit.
- **Probability Threshold**: Adjust the threshold to filter predictions based on confidence levels.

### Model Details:
- **Architecture**: LeNet5, a convolutional neural network designed for image classification.
- **Input Size**: The model expects grayscale images of size 28x28 pixels.

### Instructions:
1. Choose how to input your digit (upload or draw).
2. Set the probability threshold for predictions.
3. The predictions are displayed with a message or warning instantaneously!

### Connect with Me:
[![LinkedIn](logos/linkedin.png)](https://www.linkedin.com/in/advait-kisar/) [LinkedIn](https://www.linkedin.com/in/advait-kisar/)

[![Kaggle](logos/kaggle.png)](https://www.kaggle.com/advaitkisar) [Kaggle](https://www.kaggle.com/advaitkisar)

[![LeetCode](logos/leetcode.png)](https://leetcode.com/u/advait_kisar/) [LeetCode](https://leetcode.com/u/advait_kisar/)

[![GitHub](logos/github.png)](https://github.com/AdvaitKisar/) [GitHub](https://github.com/AdvaitKisar/)

Thank you for using this web app!
""")

model = load_model()
st.write("""
        # Single Digit Recognition
        """
        )

option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Draw a Doodle'))
if option == "Upload Image File":
  file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])
  if file is not None:
    image = Image.open(file)
    w, h = image.size
    if w != h:
      crop = transforms.CenterCrop(min(w, h))
      image = crop(image)
      wnew, hnew = image.size
      print(wnew, hnew)
    # st.image(image, width=500, caption="Image of the digit")
    threshold = st.slider("Set the probability threshold:", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    import_and_predict(image, model, threshold)
    common_message()
elif option == "Draw a Doodle":
  st.markdown("""
  Draw on the canvas to recognise the digit!
  """)
  st.write("Note: Draw the image such that the digit occupies majority of the canvas and is centered in the canvas.")

  # Fixed brush parameters
  b_width = 10  # Fixed brush width
  b_color = "#000000"  # Black ink color
  bg_color = "#FFFFFF"  # White background color

  col1, col2 = st.columns(2)

  with col1:
    # Create a canvas component
    canvas = st_canvas(
      stroke_width=b_width,
      stroke_color=b_color,
      background_color=bg_color,
      update_streamlit=True,
      height=300,
      width=300,
      key="canvas",
  )
  with col2:
    image = canvas.image_data
    # Do something interesting with the image data
    if image is not None:
        image = Image.fromarray(image)
        w, h = image.size
        if w != h:
          crop = transforms.CenterCrop(min(w, h))
          image = crop(image)
          wnew, hnew = image.size
          print(wnew, hnew)
        # st.image(image, width=500, caption="Image of the digit")
        threshold = st.slider("Set the probability threshold:", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        import_and_predict(image, model, threshold)

  common_message()
