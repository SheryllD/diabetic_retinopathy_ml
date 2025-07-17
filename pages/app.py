#!/usr/bin/env python
# coding: utf-8
# # Libraries
import streamlit as st
st.set_page_config(page_title="Diabetes Retinopathy Chatbot", page_icon="👁️")

from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import datetime
from chatbot import run_chatbot
from detect import run_detection
from tensorflow.keras.models import load_model


#Set up
with st.sidebar:
    st.header("Diabetes Retinopathy")
    choice = st.sidebar.radio("Choose a page:", 
                              ("About The Project", "Diabetes Retinopathy Detection", "💬 Chatbot"))

# Routing Logic   
if choice.startswith("Diabetes"):
    run_detection()
elif choice.startswith("💬"): 
    run_chatbot()
elif choice == "About The Project":
    # Path to about.md(fyi, one level up from pages)
    about_md_path = os.path.join(os.path.dirname(__file__), "..", "about.md")
    with open(about_md_path, "r", encoding="utf-8") as f:
        about_content = f.read()
    st.title("About the Project:")
    st.markdown(about_content, unsafe_allow_html=True)

