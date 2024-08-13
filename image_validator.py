import streamlit as st
import numpy as np
import cv2
from glob import glob
import requests
import base64
from PIL import Image
from io import BytesIO
import json

def calculate_blurriness(im):
    return cv2.Laplacian(im, cv2.CV_64F).var()

def create_laplacian_image(im):
	laplacian = cv2.Laplacian(im, cv2.CV_64F)
	laplacian = (laplacian/laplacian.max())
	laplacian = np.array(laplacian * 255, dtype = np.uint8)	
	return laplacian

def calculate_brightness(im):
    return np.average(im)

def remove_blurriness(im):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(im, -1, sharpen_kernel)

regular_images = glob("../regular_images/*.jpg")
blurry_images = glob("../blurry_images/*.jpg")
uploaded_images = glob("../upload/*.jpg")

image_dict = {}
for image in uploaded_images:
    image_dict["Uploaded " + image.split('\\')[1]] = image

for image in regular_images:
    image_dict[image.split('\\')[1][:-4]] = image

for image in blurry_images:
    image_dict["blurry " + image.split('\\')[1][:-4]] = image
    
st.title("Image Quality")

option = st.selectbox("Select image", options=image_dict.keys())
st.image(image_dict[option],"Original Image")

im = cv2.imread(image_dict[option])

api_online = True

if api_online:
    tab1, tab2 = st.tabs(["Metrics", "Blur Analysis"])
    blurriness = calculate_blurriness(im)
    brightness = calculate_brightness(im)
    laplacian = create_laplacian_image(im)

    if blurriness > 100:
        text = "Not Blurry"
        
    else:
        text = f"{np.power((100 - blurriness)/100,2)*100:.1f}% Blurry"

    with tab1:
        st.write(text)
        st.write(f"Brightness: {brightness*100/255:.1f}%")

    with tab2:
        st.image(laplacian,"Visualization of blurriness")

        if blurriness <= 100:
            cv_image = cv2.imread(image_dict[option])
            unblurred_image = remove_blurriness(cv_image)
            unblurred_image = cv2.cvtColor(unblurred_image, cv2.COLOR_BGR2RGB)
            st.image(unblurred_image, "Attempted de-blurring of image")