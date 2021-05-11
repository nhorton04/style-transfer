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

    st.image(os.path.abspath(os.getcwd()) + '/images/styletransfertext.png')

    folder = os.path.abspath(os.getcwd())
    folder = folder + '/models/good'

    exp_folder = os.path.abspath(os.getcwd())
    exp_folder = exp_folder + '/models/experimental'

    decent_folder = os.path.abspath(os.getcwd())
    decent_folder = decent_folder + '/models/decent'

    image_folder = os.path.abspath(os.getcwd())
    image_folder = image_folder + '/images/styles'

    fnames = []
    imgnames = []
    decentfs = []
    expfs = []

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

    for basename in os.listdir(image_folder):
        imgname = os.path.join(image_folder, basename)
        imgnames.append(imgname)

    uploaded_file = st.file_uploader(
        "Choose an image - works better with smaller files", type=['jpg', 'png', 'webm', 'mp4', 'gif', 'jpeg'])

    if uploaded_file is None:
        uploaded_file = os.path.abspath(os.getcwd()) + '/images/pence.jpeg'

    tiers = ['good', 'decent', 'experimental']

    choice = st.selectbox('Select the tier of model to choose from:', tiers)

    if choice == 'good':
        checkpoint = st.selectbox('Select a good model', fnames)
    elif choice == 'decent':
        checkpoint = st.selectbox('Select a decent model', decentfs)
    elif choice == 'experimental':
        checkpoint = st.selectbox('Select an experimental model', expfs)

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

    print(real_name)

    if real_name is not None:
        try:
            st.image(uploaded_file, width=400)
        except:
            st.markdown(
                "![Didn't display : (]({uploaded_file})")
        st.write('+')

        try:
            st.image(image_folder + '/' + real_name + '.jpg', width=400)
        except:
            try:
                st.image(image_folder + '/' + real_name + '.jpeg', width=400)
            except:
                try:
                    st.image(image_folder + '/' +
                             real_name + '.webp', width=400)
                except:
                    pass
                pass
            pass

        st.write('=')
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
        st.image(f"images/outputs/stylized-{fn}", width=640)

    except:
        pass


if __name__ == "__main__":
    main()
