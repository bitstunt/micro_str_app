import streamlit as st
from modules import generate_superpixels

sidebar = st.sidebar
sidebar.slider(":blue[Scale]", min_value=0, max_value=500, value=300, step=1, key="scale")
sidebar.slider(":blue[Sigma]", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="sigma")

data = st.file_uploader('Upload your microstructure')

if data is not None:
    col1, col2 = st.columns(2)
    col1.image(data)
    col1.write("Original Image")
    col2.image(generate_superpixels(data, st.session_state.scale, st.session_state.sigma))
    col2.write("Superpixels marked by boundaries")
