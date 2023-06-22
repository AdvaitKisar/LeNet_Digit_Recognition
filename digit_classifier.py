import streamlit as st
import torch

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model = LeNet5()
  model.load_state_dict(torch.load('./model.pth'))
  return model

model = load_model()
st.write("""
        # Digit Classification
        """
        )

file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])

import torch.transforms as transforms
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
  string = f"The model predicts the image as yhat = {yhat} with {p:.2f} % probability."
  st.success(string)

from PIL import Image
if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  import_and_predict(image, model)
