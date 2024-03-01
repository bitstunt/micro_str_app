import streamlit as st

data = st.file_uploader('Upload your microstructure')

if data is not None:
    image = data.getvalue()
    st.image(image)