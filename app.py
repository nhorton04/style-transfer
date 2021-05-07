import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import tqdm
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

    uploaded_file = st.file_uploader(
        "Choose an image - works better with smaller files", type=['jpg', 'png', 'webm', 'mp4', 'gif', 'jpeg'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
    else:
        uploaded_file = os.path.abspath(os.getcwd()) + '/images/pence.jpeg'
        st.image(uploaded_file, width=200)

    folder = os.path.abspath(os.getcwd())
    folder = folder + '/models'

    fnames = []

    for basename in os.listdir(folder):
        fname = os.path.join(folder, basename)
        if fname.endswith('.pth'):
            fnames.append(fname)
            print(f'adding:{fname}')
    print(fnames)
    checkpoint = st.selectbox('Select a pretrained model', fnames)

    checkpoint_image = str(checkpoint)
    print(checkpoint_image)

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

        fn = str(np.random.randint(0, 100)) + 'image.jpg'
        save_image(stylized_image, f"images/outputs/stylized-{fn}")

        st.image(f"images/outputs/stylized-{fn}")
        st.write(checkpoint)

    except:
        pass


if __name__ == "__main__":
    main()
