import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model1=load_model('BrainTumor10epochs.keras')

image=cv2.imread('C:\\Users\\Lenovo\\Downloads\\brain tumor\\pred\\pred60.png')

img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_image=np.expand_dims(img,axis=0)
result = model1.predict(input_image)
if(result==[[0.]]):
    print("No Brain Tumor")
else:
    print("Brain Tumor is present")


