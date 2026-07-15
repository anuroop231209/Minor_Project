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
        experiences = analysis.get('work_experience', [])
        if not experiences:
            st.markdown("NA")
            return
        seen = set()
        for exp in experiences:
            job_title = exp.get('job_title', 'N/A')
            company = exp.get('company', 'N/A')
            start_date = exp.get('start_date', 'N/A')
            end_date = exp.get('end_date', 'present')
            resp = exp.get('responsibilities', '')

            if isinstance(resp, list):
                desc = '' .join(str(item) for item in resp).strip() or "No detailed responsibilities provided. provided."
            else:
                desc = str(resp).strip() or "No detailed responsibilities provided."

            key = f"{job_title}_{company}_{start}-{end}"
            if key  not in seen:
                seen.add(key)
                with st.expander(f"{job_title} at {company}" ,expanded=false):
                    col1, col2 = st.columns(1,3)
                    with col1:
                        st.markdown(f"**Period(time period):** \n{start} - {end}")
                    with col2:
                        st.markdown(f"**Responsibilities:** \n{desc}")
                st.markdown("---")  # Add a horizontal line between experiences
               

def display_wordcloud(resume_text):
    """Generate and display word cloud wth improved styling."""
with stylable_container(
        key="wordcloud",
        css_style="""{
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            background: white;
       } """
       ):
        st.subheader("Resume Keywords cloud")

        if resume_text.strip():
            with st.spinner("Generating word cloud..."):
                wc_img = generate_wordcloud(resume_text)
                st.image(wc_img, caption=" Most frequent keywords from your resume", use_column_width=True)

        else:
            st.warning("No text found in the resume to generate a word cloud.")

        
def process_resume(file_path: str) -> Tuple[Optional[dict], Optional[ str]]:
    """Process the resume and return analysis or error ."""
    with st.spinner("Analyzing resume content.."):
        resume_text = extract_text_from_file(file_path)

        #if resume text is empty, return error or too short shoow error 

        if not resume_text or len(resume_text.strip()) < 30:
         st.error("Could not extract usable text from this file. If this is a scanned image or low-quality photo, please upload a higher quality scan or a text-based PDF.")
        st.info(f"Extracted text (debug, first 1000 chars):\n{resume_text[:1000]}")
        

        return None, "Text extraction failed: insufficient content."
        
        # Reset extraction attempts counter
        st.session_state.extraction_attempts = 0
        
        # Try extraction with progressively more detailed prompts
        analysis = None
        error = None
        
        while not analysis and st.session_state.extraction_attempts < 3:
            st.session_state.extraction_attempts += 1
            analysis = llama3_extract_resume_info(resume_text, st.session_state.extraction_attempts)

            # Improved robust JSON extraction using extract_first_json and fallback cleaning
            if not isinstance(analysis, dict):
                import json, re
                from model.model import extract_first_json
                parsed = extract_first_json(analysis)
                if parsed is not None:
                    analysis = parsed
                else:
                    #try to extravt the largest JSON Substring
                    json_start = analysis.find('{')
                    json_end = analysis.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = analysis[json_start:json_end + 1]
                        json_str = json_str.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

                        json_str = re.sub(r',\s*}', '}', json_str)  # Remove newlines, tabs, etc.
                        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before closing brackets
                        try:
                            analysis = json.loads(json_str)
                        except Exception :
                            st.warning("Raw model output (for debugging):\n{analysis}")
                            return {"Raw_output": str(analysis)}, "JSON parsing failed: Could not parse model output into valid JSON."
                    else:
                        try:
                            analysis = json.loads(analysis)
                        except Exception:
                            st.warning(f"Raw model output (for debugging):\n{analysis}")
                            return {"raw_output": str(analysis)}, "Could not parse structured data from model output"
            if 'raw_output' in analysis:
                st.warning(f"Raw model output (for debugging):\n{analysis}")
                return analysis, "Could not parse structured data from model output"
            
            # Validate required fields
            required_fields = ['name', 'email', 'phone', 'skills', 'experiences']
            missing_fields = [field for field in required_fields if not analysis.get(field)]

            # If any required fields are missing, fill them with N/A or [] and proceed with a warning
            if missing_fields:
                for field in missing_fields:
                    if field in ['skills', 'experiences', 'degree', 'universities']:
                        analysis[field] = []
                    else:
                        analysis[field] = "N/A"
                st.warning(f"Some information could not be extracted from your resume: {', '.join(missing_fields)}. These fields have been set to N/A or left blank.")
                break

            # If we have a valid analysis, break the loop
            if analysis:
                break
                
        if not analysis:
            return None, error or "Resume analysis failed after multiple attempts"
            
        # Calculate score and save to DB
        st.session_state.resume_score = calculate_resume_score(analysis)
        st.session_state.analysis_done = True
        
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        
        insert_resume_data(
            name=analysis.get('name', 'N/A'),
            email=analysis.get('email', 'N/A'),
            score=st.session_state.resume_score,
            timestamp=timestamp,
            candidate_level=analysis.get('cand_level', 'N/A'),
            Experience=str(analysis.get('experiences', [])),
            Skill=str(analysis.get('skills', [])),
        )
        
        return analysis, None
    
def calculate_resume_score(analysis: dict) -> float:
    """Calculate resume score from analysis using degree, skills, and experience."""
    # If all key fields are NA/empty, return 0
    degree = analysis.get('degree', '')
    skills = analysis.get('skills', [])
    experiences = analysis.get('experiences', [])
    degree_na = (not degree or (isinstance(degree, str) and degree.strip().lower() in ['n/a', 'na', 'none', 'null', '']) or (isinstance(degree, list) and not any(str(d).strip() and str(d).strip().lower() not in ['n/a', 'na', 'none', 'null', ''] for d in degree)))
    skills_na = (not skills or (isinstance(skills, str) and skills.strip().lower() in ['n/a', 'na', 'none', 'null', '']) or (isinstance(skills, list) and not any(str(s).strip() and str(s).strip().lower() not in ['n/a', 'na', 'none', 'null', ''] for s in skills)))
    experiences_na = (not experiences or (isinstance(experiences, list) and not experiences))
    if degree_na and skills_na and experiences_na:
        return 0.0
    score_text = str(analysis.get('summary', '')) + ' '
    # Add degree
    if degree and degree != 'N/A':
        score_text += str(degree) + ' '
    # Add skills
    if isinstance(skills, list):
        score_text += ' '.join([str(skill) for skill in skills if skill]) + ' '
    elif isinstance(skills, str) and skills.strip():
        score_text += skills.strip() + ' '
    # Add experience responsibilities
    responsibilities_list = []
    for exp in experiences:
        resp = exp.get('responsibilities', '')
        if isinstance(resp, list):
            responsibilities_list.append(' '.join(str(item) for item in resp))
        else:
            responsibilities_list.append(str(resp))
    score_text += ' '.join(responsibilities_list)
    return bilstm_score_resume(score_text) * 100

            





    
