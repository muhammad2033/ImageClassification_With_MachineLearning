import numpy as np 
import streamlit as st 
import pickle
from keras.utils import load_img, img_to_array
from skimage.feature import hog 

with open('HOG.pkl','rb') as m:
    model = pickle.load(m)
classes = ['car', 'cricketBall', 'IceCream Cone']
st.title('Image Classification App')
uploader = st.file_uploader('Select Image', type=['jpg', 'jpeg', 'png'])

if uploader is not None:
    img = load_img(uploader, target_size=(100,100), color_mode='grayscale')
    imgArr = img_to_array(img)
    img2D = imgArr.squeeze(2)
    res = model.predict(img2D)
    st.write(classes[res[0]]); st.image(img)