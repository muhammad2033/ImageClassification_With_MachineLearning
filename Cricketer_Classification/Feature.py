import streamlit as st
import numpy as np
import cv2
import pickle
from keras.utils import load_img, img_to_array

with open('HOG.pkl', 'rb') as mo:
    loadedModel = pickle.load(mo)

st.title("RandomForest.pkl")
classes = ['ABD', 'SAM']

st.title('Image classification')
img = st.file_uploader('Select image', type=['jpg', 'png', 'jpeg'])

if img is not None:
    img = load_img(img, target_size=(100, 100), color_mode='grayscale')
    imgarr = img_to_array(img)
    imgflatten = imgarr.flatten()
    st.write('Flatten shape is ', imgflatten.shape)
    res = loadedModel.predict(imgflatten.reshape(1, -1))
    st.write(f'Model predicted class is: {classes[int(res[0])]}')
    st.image(img, caption='Uploaded Image', width=300)
