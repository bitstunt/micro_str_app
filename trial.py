import streamlit as st
from PIL import Image
from numpy import asarray
from modules import generate_superpixels, run

sidebar = st.sidebar
sidebar.slider(":blue[Scale]", min_value=0, max_value=500, value=300, step=1, key="scale")
sidebar.slider(":blue[Sigma]", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="sigma")

data = st.file_uploader('Upload your microstructure')

if data is not None:
    col1, col2 = st.columns(2)
    col1.image(data)
    col1.write("Original Image")
    PIL_Image = Image.open(data)
    image = asarray(PIL_Image)
    with st.spinner('Getting Superpixels'):
        superpixels, seg_map = generate_superpixels(image, st.session_state.scale, st.session_state.sigma)
        col2.image(superpixels)
        col2.write("Superpixels marked by boundaries")
    st.button("Segment Image", on_click=st.write("Hi"))
