import streamlit as st
import numpy as np
import cv2
import pickle
from keras.utils import load_img, img_to_array
from skimage.feature import local_binary_pattern

# Load multiple models
model_paths = {
    'SVM': 'SVM.pkl',
    'LBP': 'LBP.pkl',
    'HOG': 'HOG.pkl',
    'Voting_Classifier': 'voting_classifier.pkl',
    'LogisticRegression': 'LogisticRegression.pkl',
    'GBC': 'GBC.pkl'
   
    # Add more models as needed
}

loaded_models = {}
for model_name, model_path in model_paths.items():
    with open(model_path, 'rb') as model_file:
        loaded_models[model_name] = pickle.load(model_file)

# Define classes for each model
classes = {
    'SVM':               ['ABD', 'SAM'],
    'LBP':               ['ABD', 'SAM'],
    'HOG':               ['ABD', 'SAM'],
    'Voting_Classifier': ['ABD', 'SAM'],
    'LogisticRegression':['ABD', 'SAM'],
    'GBC':               ['ABD', 'SAM']
   
    # Add more classes for more models
}

# Create a sidebar for model selection
selected_model = st.sidebar.selectbox('Select Model', list(model_paths.keys()))

st.title('Image Classification')

# Add a header with model information
st.header(f"Selected Model: {selected_model}")

img = st.file_uploader('Select image', type=['jpg', 'png', 'jpeg'])
# Import additional libraries for Gabor filtering

# ...

if img is not None:
    img = load_img(img, target_size=(100, 100), color_mode='grayscale')
    img_arr = img_to_array(img)
    
    if selected_model == 'HOG':
        img_flatten = img_arr.squeeze(2)
    elif selected_model == 'LBP':
        lbp = local_binary_pattern(img_arr, 3, 1).squeeze(2)
    else:
        img_flatten = img_arr.flatten()

    st.write('Flatten shape is ', img_flatten.shape)

    # Use the selected model for prediction
    selected_loaded_model = loaded_models[selected_model]
    res = selected_loaded_model.predict([img_flatten])
    predicted_class = classes[selected_model][res[0]]
    st.write(f'{selected_model} predicted class is: {predicted_class}')

    # Display the uploaded image
    st.image(img, caption="Uploaded Image")
