import os
import time
import datetime
import base64
import streamlit as st
from streamlit_lottie import  st_lottie

from typing import optional , Tuple, Dict, Any
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from streamlit_extras.stylable_container import stylable_container

from backend import (
    hash_password, hash_file,extract_text_from_file,
    generate_wordcloud,get_table_download_link,
    insert_resume_data, get_resume_score,
    validate_user,register_user,get_candidate_data,
    get_stats
)
from model import llama3_extract_resume_info, bilstm_score_resume

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def gradient_text(text,color1,color2):
    return f"""
    <style>
    .gradient-text {{
        background: -webkit-linear-gradient({color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;

    }}
    </style>
    <span class="gradient-text">{text}</span>
    """

def initialize_session_state():
    """Initialize all session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'candidate_logged_in' not in st.session_state:
        st.session_state.candidate_logged_in = False
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    if 'candidate_username' not in st.session_state:
        st.session_state.candidate_username = ''
    if 'resume_score' not in st.session_state:
        st.session_state.resume_score = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'resume_score' not in st.session_state:
        st.session_state.resume_score = None
    if 'extraction_attempts' not in st.session_state:
        st.session_state.extraction_attempts = 0

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save the uploaded file to the 'uploads' directory."""
    try :
        os.makedirs('Uploaded_resume', exist_ok=True)
        save_path = os.path.join('Uploaded_resume', uploaded_file.name)

        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return save_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")

    return None

def display_candidate_info(analysis):
    """Dsiplay candidate infromation (basic) with improved styling """
    with Stylable_container(
        key="candidate_info",
        css_style="""
            border: 2px solid #4CAF50;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            margin-bottom: 1.5rem;
            background: white;
        """
    ):
        