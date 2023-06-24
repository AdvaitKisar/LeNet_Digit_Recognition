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

def import_and_predict(img, model):
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
  yhat = int(float(yhat.numpy()[0]))
  string = f"The uploaded image is of the digit {yhat} with {p:.2f} % probability."
  st.success(string)
  st.write("Thanks for using this classifier.")
  st.write("Made by Advait Amit Kisar.")
  st.write("Reach out to me for any queries/discussion at +91 7774035501 or advaitkisar2509@gmail.com.")


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = LeNet5()
  model.load_state_dict(torch.load('./model.pth'))
  return model

model = load_model()
st.write("""
        # Digit Recognition
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
    st.image(image, width=500, caption="Image of the digit")
    import_and_predict(image, model)
elif option == "Draw a Doodle":
  st.markdown("""
  Draw on the canvas, to recognise the digit!
  """)
  st.sidebar.header("Configuration")

  # Specify brush parameters and drawing mode
  b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
  b_color = st.sidebar.color_picker("Enter brush color hex: ")
  bg_color = st.sidebar.color_picker("Enter background color hex: ", "#FFFFFF")
  drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

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
      st.image(image, width=500, caption="Image of the digit")
      import_and_predict(image, model)
