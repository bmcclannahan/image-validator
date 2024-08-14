import streamlit as st
import numpy as np
import cv2
from glob import glob
from PIL import Image, ImageDraw
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os

if 'cfg' not in st.session_state:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.OUTPUT_DIR = "detectron_files"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    predictor = DefaultPredictor(cfg)

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

def detect_tires(im):
    outputs = predictor(im)
    outputs = outputs["instances"].get_fields()
    output_dict = {
        "pred_boxes": outputs["pred_boxes"].tensor.numpy().tolist(),
        "scores": outputs["scores"].numpy().tolist(),
        # "pred_classes": outputs["pred_classes"].numpy().tolist(),
        # "pred_masks": outputs["pred_masks"].numpy().tolist()
    }
    return output_dict

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

api_online = True

if api_online:
    tab1, tab2, tab3 = st.tabs(["Metrics", "Blur Analysis", "Tire Detection"])
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

    tire_dict = detect_tires(im)
    boxes = tire_dict["pred_boxes"]
    scores = tire_dict["scores"]

    with tab3:
        if len(boxes) > 0:
            im = Image.open(image_dict[option])
            for i in range(len(boxes)):
                draw = ImageDraw.Draw(im)
                draw.rectangle(boxes[i], outline='red', width=3)
            st.image(im)
        else:
            st.write("No Tires Detected")