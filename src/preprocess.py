import tensorflow as tf
import numpy as np
from PIL import Image

IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(image: Image.Image):
     # Force RGB (important!)
  image = image.convert("RGB")
   # Resize to training size
  image = image.resize((IMG_WIDTH, IMG_HEIGHT)) 
  img = np.array(image) 
  img = np.expand_dims(img, axis=0) 
  return img

