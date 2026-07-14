import os
import time
import datetime
import base64
import streamlit as st
from streamlit_lottie import  st_lottie

from typing import Optional , Tuple, Dict, Any
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
    with stylable_container(
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
        st.subheader("Candidate Information")
        name = analysis.get('name', 'N/A')
        email = analysis.get('email', 'N/A')
        phone = analysis.get('phone', 'N/A')
        col1, col2 ,col3 = st.columns(3)
        with col1:
            st.markdown(f"**Name:** ")
            st.info(name if name.strip() else "N/A")
        with col2:
            st.markdown(f"**Email:** ")
            st.info(email if email.strip() else "N/A")
        with col3:
            st.markdown(f"**Phone:** ")
            st.info(phone if phone.strip() else "N/A")
        
def display_skills(analysis):
    """Display skills section above work experience."""
    with stylable_container(
        key="skills_section",
        css_style="""
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            background: white;
        """
    ):
        st.subheader("Skills")
        skills = analysis.get('skills', [])
        if not skills or(isinstance(skills, str) and not skills.strip()):
            st.markdown("NA")
        else:
            if isinstance(skills, str):
                import ast
                try:
                    skills = ast.literal_eval(skills)
                except Exception as e:
                    skills = [skills]
            if isinstance(skills, list):
                st.markdown(", ".join([skill.strip() for skill in skills if skill]) or "NA")

            else:
                st.markdown(str(skills) if str(skills).strip() else "NA")

def display_degree(analysis):
    """Display degree section above work experience."""
    with stylable_container(
        key="degree_section",
        css_style="""
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            background: white;
        """
    ):
        st.subheader("Degree")
        degree = analysis.get('degree', 'N/A')
        #if degree is string , try to parse as list if it looks like a list
        if isinstance(degree, str):
            import ast
            try:
                degree_eval = ast.literal_eval(degree)
                if isinstance(degree_eval, list):
                    degree = degree_eval
                   
            except Exception:
                pass
        #Display logic for list or string 
        if not degree or (isinstance(degree, str) and not degree.strip().lower() in ['na', 'none', 'null','']):
            st.markdown("NA")
        elif isinstance(degree, list):
            degree_list = [str(d) for d in degree if d and str (d).strip().lower() not in ['na', 'none', 'null',''] ]
            st.markdown("**Degrees:** " + ", ".join(degree_list) if degree_list else "NA")
        else:
            st.markdown("**Degree:** " + {degree  if str(degree).strip() else "NA"})


def display_work_experience(analysis):
    """ Display work experience section with improved styling."""
    with stylable_container(
        key="work_exp",
        css_style={"""
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            background: white;
        """
        }
        
        ):
        st.subheader("Work Experience")
        
    
