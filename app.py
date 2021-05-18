import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
# import tqdm
import streamlit as st
from models import TransformerNet
from utils import *
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import tkinter as tk
import os
import cv2
import matplotlib
matplotlib.use('agg')


def main():
    st.set_page_config(layout="wide")

    st.image(os.path.abspath(os.getcwd()) + '/images/styletransfertext.png')

    folder = os.path.abspath(os.getcwd())
    folder = folder + '/models/good'

    exp_folder = os.path.abspath(os.getcwd())
    exp_folder = exp_folder + '/models/experimental'

    decent_folder = os.path.abspath(os.getcwd())
    decent_folder = decent_folder + '/models/decent'

    new_folder = os.path.abspath(os.getcwd())
    new_folder = new_folder + '/models/new'

    image_folder = os.path.abspath(os.getcwd())
    image_folder = image_folder + '/images/styles'

    fnames = []
    imgnames = []
    decentfs = []
    expfs = []
    newf = []

    for basename in os.listdir(folder):
        fname = os.path.join(folder, basename)
        if fname.endswith('.pth'):
            fnames.append(fname)

    for basename in os.listdir(decent_folder):
        fname = os.path.join(decent_folder, basename)
        if fname.endswith('.pth'):
            decentfs.append(fname)

    for basename in os.listdir(exp_folder):
        fname = os.path.join(exp_folder, basename)
        if fname.endswith('.pth'):
            expfs.append(fname)

    for basename in os.listdir(new_folder):
        fname = os.path.join(new_folder, basename)
        if fname.endswith('.pth'):
            newf.append(fname)

    print(newf)

    for basename in os.listdir(image_folder):
        imgname = os.path.join(image_folder, basename)
        imgnames.append(imgname)

    uploaded_file = st.file_uploader(
        "Choose an image - a dataset of 20,000 celebrity faces was used to train the models, so pictures of faces will have best results", type=['jpg', 'png', 'webm', 'mp4', 'gif', 'jpeg'])

    if uploaded_file is None:
        uploaded_file = os.path.abspath(os.getcwd()) + '/images/pence.jpeg'

    tiers = ['good', 'decent', 'experimental', 'new']

    choice = st.selectbox('Select the tier of model to choose from:', tiers)

    if choice == 'good':
        checkpoint = st.selectbox('Select a good model', fnames)
    elif choice == 'decent':
        checkpoint = st.selectbox('Select a decent model', decentfs)
    elif choice == 'experimental':
        checkpoint = st.selectbox('Select an experimental model', expfs)
    elif choice == 'new':
        checkpoint = st.selectbox('Select an new model', newf)

    checkpoint_image = str(checkpoint)

    if choice == 'good':
        image_name = checkpoint_image.rsplit(
            'good/', 1)
        real_name = image_name[1].rsplit(".pth")[0]
    elif choice == 'decent':
        image_name = checkpoint_image.rsplit(
            'decent/', 1)
        real_name = image_name[1].rsplit(".pth")[0]
    elif choice == 'experimental':
        image_name = checkpoint_image.rsplit(
            'experimental/', 1)
        real_name = image_name[1].rsplit(".pth")[0]
    elif choice == 'new':
        image_name = checkpoint_image.rsplit(
            'new/', 1)
        real_name = image_name[1].rsplit(".pth")[0]
        even_realer = real_name.rsplit("_")[0]

    # print(even_realer)
    print(f'real_name: {real_name}')

    col1, col2, col3, col4, col5 = st.beta_columns((1, .1, 1, .1, 2))
    # col1, col3, col5 = st.beta_columns((1, 1, 2))

    col1.header("Content")
    col1.image(uploaded_file, use_column_width=True)

    col2.header('+')

    col3.header("Style")

    try:
        col3.image(image_folder + '/' + real_name +
                   '.jpg', use_column_width=True)
    except:
        try:
            col3.image(image_folder + '/' + real_name +
                       '.jpeg', use_column_width=True)
        except:
            try:
                col3.image(image_folder + '/' +
                           real_name + '.webp', use_column_width=True)
            except:
                try:
                    col3.image(image_folder + '/' +
                               real_name + '.png', use_column_width=True)
                except:
                    pass
                pass
            pass
        pass

    try:
        col3.image(image_folder + '/' + even_realer +
                   '.jpg', use_column_width=True)
    except:
        try:
            col3.image(image_folder + '/' + even_realer +
                       '.jpeg', use_column_width=True)
        except:
            try:
                col3.image(image_folder + '/' +
                           even_realer + '.webp', use_column_width=True)
            except:
                try:
                    col3.image(image_folder + '/' +
                               even_realer + '.png', use_column_width=True)
                except:
                    pass
                pass
            pass
        pass

    col4.header('=')

    img_name = [i for i in imgnames if real_name in i]
    print(f'image is: {img_name}')

    os.makedirs("images/outputs", exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(
        checkpoint, map_location='cpu'))
    # transformer.load_state_dict(torch.load(checkpoint))
    transformer.eval()

    # Prepare input
    try:
        image_tensor = Variable(transform(Image.open(
            uploaded_file).convert('RGB'))).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()

        # fn2 = str(np.random.randint(0, 100)) + 'image.gif'
        fn = str(np.random.randint(0, 100)) + 'image.jpg'
        save_image(stylized_image, f"images/outputs/stylized-{fn}")

        # if uploaded image is gif / video
        # st.markdown(
        #     f"![Alt Text](images/outputs/stylized-{fn2})")
        col5.image(f"images/outputs/stylized-{fn}", use_column_width=True)

    except:
        pass


if __name__ == "__main__":
    main()
