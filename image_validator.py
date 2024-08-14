import streamlit as st
import numpy as np
import cv2
from glob import glob
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

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

uploaded_images = glob("upload/*.jpg")

image_dict = {}
for image in uploaded_images:
    if '\\' in image:
        image_dict[image.split('\\')[1]] = image
    else:
        image_dict[image.split('/')[1]] = image
    
st.title("Image Quality")

option = st.selectbox("Select image", options=image_dict.keys())
st.image(image_dict[option],"Original Image")

im = cv2.imread(image_dict[option])
open_cv_image = im[:, :, ::-1].copy()
gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

api_online = True

if api_online:
    tab1, tab2 = st.tabs(["Metrics", "Blur Analysis"])
    blurriness = calculate_blurriness(gray)
    brightness = calculate_brightness(gray)
    laplacian = create_laplacian_image(gray)

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