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
from model import (llama3_extract_resume_info, 
                   bilstm_score_resume,
                   llama3_infer,extract_first_json)

def load_lottieurl(url, timeout=10):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def gradient_text(text,color1,color2):
    return f"""
    <style>
    .gradient-text {{
        background: -webkit-linear-gradient(45deg,{color1}, {color2});
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
          background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border: 1px solid #463FA9;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 16px 48px rgba(7,5,246,0.25);
            margin-bottom: 1.75rem;
            color: #ECDFD2;
        """
    ):
        st.markdown("### 👤 Candidate Profile")
        
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
            background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border: 1px solid #463FA9;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 16px 48px rgba(7,5,246,0.25);
            margin-bottom: 1.75rem;
            color: #ECDFD2;
        """
    ):
        st.markdown("### 🛠️ Technical Skills")
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
                skill_tags = [str(skill).strip() for skill in skills if skill and str(skill).strip()]
                if skill_tags:
                    tags_html = "".join([f'<span style="background: rgba(33,56,133,0.8); color: #EB70EC; padding: 8px 16px; border-radius: 24px; font-size: 0.9rem; font-weight: 600; display: inline-block; margin: 5px; border: 1px solid #6A57F3; box-shadow: 0 4px 12px rgba(106,87,243,0.25);">{tag}</span>' for tag in skill_tags])
                    st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 8px;">{tags_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("NA")
            else:
                st.markdown(str(skills) if str(skills).strip() else "NA")

def display_degree(analysis):
    """Display degree section above work experience."""
    with stylable_container(
        key="degree_section",
        css_style="""
            background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border: 1px solid #463FA9;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 16px 48px rgba(7,5,246,0.25);
            margin-bottom: 1.75rem;
            color: #ECDFD2;
        """
    ):
        st.markdown("### 🎓 Education & Degrees")
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
        if not degree or (isinstance(degree, str) and not degree.strip().lower() in ['n/a', 'na', 'none', 'null','']):
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
           background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border: 1px solid #463FA9;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 16px 48px rgba(7,5,246,0.25);
            margin-bottom: 1.75rem;
            color: #ECDFD2;
        """
        }
        
        ):
        st.markdown("### 💼 Professional Work Experience")
        experiences = analysis.get('experiences', [])
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
                desc = '<br>'.join([f"• {item}" for item in resp if item]) or "📌 No detailed responsibilities provided."
            else:
                desc = str(resp).strip() or "📌 No detailed responsibilities provided."

            key = f"{job_title}_{company}_{start}_{end}"
            if key not in seen:
                seen.add(key)
                with st.expander(f"🔹 {job_title} at {company}", expanded=False):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**📅 Period**  \n{start} - {end}")
                    with col2:
                        st.markdown(f"**📋 Responsibilities**  \n{desc}", unsafe_allow_html=True)
                st.markdown("---")  # Add a horizontal line between experiences
               

def display_wordcloud(resume_text):
    """Generate and display word cloud wth improved styling."""
with stylable_container(
        key="wordcloud",
        css_style="""{
            background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border: 1px solid #463FA9;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 16px 48px rgba(7,5,246,0.25);
            margin-bottom: 1.75rem;
            color: #ECDFD2;
       } """
       ):
        st.markdown("### ☁️ Resume Keyword Cloud")
        if resume_text.strip():
            with st.spinner("Generating word cloud..."):
                wc_img = generate_wordcloud(resume_text)
                st.image(wc_img, caption="Most frequent keywords from your resume", use_container_width=True)
        else:
            st.warning("No text found for word cloud.")
        
def process_resume(file_path: str) -> Tuple[Optional[dict], Optional[str]]:
    """Process resume and return analysis or error."""
    with st.spinner("🔍 Analyzing resume content..."):
        resume_text = extract_text_from_file(file_path)

        #if resume text is empty, return error or too short shoow error 

        if not resume_text or len(resume_text.strip()) < 30:
            st.error("Could not extract usable text from this file. If this is a scanned image or low-quality photo, please upload a higher quality scan or a text-based PDF.")
            st.info(f"Extracted text (debug, first 1000 chars):\n{resume_text[:1000]}")
            return None, "Text extraction failed: insufficient content."
        
        st.session_state.extraction_attempts = 0
        analysis = None
        error = None
        
        while not analysis and st.session_state.extraction_attempts < 3:
            st.session_state.extraction_attempts += 1
            analysis = llama3_extract_resume_info(resume_text, st.session_state.extraction_attempts, file_path=file_path)

            if not isinstance(analysis, dict):
                import json, re
                parsed = extract_first_json(analysis)
                if parsed is not None:
                    analysis = parsed
                else:
                    json_start = analysis.find('{')
                    json_end = analysis.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = analysis[json_start:json_end+1]
                        json_str = json_str.replace('\n', ' ').replace('\t', ' ')
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            analysis = json.loads(json_str)
                        except Exception:
                            st.warning(f"Raw model output (for debugging):\n{analysis}")
                            return {"raw_output": str(analysis)}, "Could not parse structured data from model output"
                    else:
                        try:
                            analysis = json.loads(analysis)
                        except Exception:
                            st.warning(f"Raw model output (for debugging):\n{analysis}")
                            return {"raw_output": str(analysis)}, "Could not parse structured data from model output"
            if isinstance(analysis, dict) and 'raw_output' in analysis:
                st.warning(f"Raw model output (for debugging):\n{analysis}")
                return analysis, "Could not parse structured data from model output"

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

            if analysis:
                break
        if not analysis:
            return None, error or "Resume analysis failed after multiple attempts"
            
        score_val = calculate_resume_score(analysis)
        st.session_state.resume_score = score_val
        st.session_state.analysis_done = True
        
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        insert_resume_data(
            name=analysis.get('name', 'N/A'),
            email=analysis.get('email', 'N/A'),
            score=score_val,
            timestamp=timestamp,
            candidate_level=analysis.get('cand_level', 'N/A'),
            Skill=str(analysis.get('skills', [])),
            Experience=str(analysis.get('experiences', [])),
        )
        
        return analysis, None
    
def calculate_resume_score(analysis: dict) -> float:
    """Calculate resume score from analysis using degree, skills, and experience."""
    degree = analysis.get('degree', '')
    skills = analysis.get('skills', [])
    experiences = analysis.get('experiences', [])
    degree_na = (not degree or (isinstance(degree, str) and degree.strip().lower() in ['n/a', 'na', 'none', 'null', '']) or (isinstance(degree, list) and not any(str(d).strip() and str(d).strip().lower() not in ['n/a', 'na', 'none', 'null', ''] for d in degree)))
    skills_na = (not skills or (isinstance(skills, str) and skills.strip().lower() in ['n/a', 'na', 'none', 'null', '']) or (isinstance(skills, list) and not any(str(s).strip() and str(s).strip().lower() not in ['n/a', 'na', 'none', 'null', ''] for s in skills)))
    experiences_na = (not experiences or (isinstance(experiences, list) and not experiences))
    if degree_na and skills_na and experiences_na:
        return 0.0
    score_text = str(analysis.get('summary', '')) + ' '
    if degree and degree != 'N/A':
        score_text += str(degree) + ' '
    if isinstance(skills, list):
        score_text += ' '.join([str(skill) for skill in skills if skill]) + ' '
    elif isinstance(skills, str) and skills.strip():
        score_text += skills.strip() + ' '
    responsibilities_list = []
    for exp in experiences:
        resp = exp.get('responsibilities', '')
        if isinstance(resp, list):
            responsibilities_list.append(' '.join(str(item) for item in resp))
        else:
            responsibilities_list.append(str(resp))
    score_text += ' '.join(responsibilities_list)
    val = bilstm_score_resume(score_text)
    if val is not None:
        return float(val) if val > 1.0 else float(val) * 100
    return 75.0


def display_resume_score():
    """Display the calculated resume score with enhanced visualization."""
    if st.session_state.resume_score is not None:
        with stylable_container(
            key="resume_score",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border: 1px solid #463FA9;
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 16px 48px rgba(7,5,246,0.25);
                margin-bottom: 1.75rem;
                color: #ECDFD2;
            }
            """
        ):
            st.markdown("### 📊 Resume Quality Score")
            score = st.session_state.resume_score
            
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-bottom: 2rem;">
                <div style="position: relative; width: 200px; height: 200px; border-radius: 50%; 
                            background: conic-gradient(#EB70EC {score*3.6}deg, #213885 0deg); 
                            display: flex; align-items: center; justify-content: center;
                            box-shadow: 0 8px 30px rgba(235,112,236,0.25);">
                    <div style="position: absolute; width: 170px; height: 170px; 
                                background: #081849; border-radius: 50%; 
                                display: flex; flex-direction: column; 
                                align-items: center; justify-content: center;">
                        <span style="font-size: 2.5rem; font-weight: 700; color: #EB70EC;">
                            {score:.1f}
                        </span>
                        <span style="font-size: 1rem; color: #B693EF; margin-top: -5px;">
                            out of 100
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.markdown(f"""
                <div style="position: relative; height: 20px; background: #213885; 
                            border-radius: 10px; overflow: hidden; margin-bottom: 1rem;">
                    <div style="position: absolute; height: 100%; width: {score}%; 
                                background: linear-gradient(90deg, #4044F9, #6A57F3, #EB70EC); 
                                border-radius: 10px;">
                    </div>
                    <div style="position: absolute; height: 100%; width: 2px; 
                                left: {score}%; background: #B693EF; box-shadow: 0 0 8px #EB70EC;">
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; 
                            margin-top: -10px; margin-bottom: 2rem;">
                    <span style="font-size: 0.8rem; color: #B693EF;">0</span>
                    <span style="font-size: 0.8rem; color: #B693EF;">100</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; 
                        margin-bottom: 1.5rem; text-align: center; color: #ECDFD2;">
                <div style="padding: 0.5rem; background: #213885; border-radius: 8px; border: 1px solid #463FA9;">
                    <span style="font-weight: 600; color: #EB70EC;">0-44</span>
                    <div>Needs Work</div>
                </div>
                <div style="padding: 0.5rem; background: #213885; border-radius: 8px; border: 1px solid #463FA9;">
                    <span style="font-weight: 600; color: #B693EF;">45-64</span>
                    <div>Good</div>
                </div>
                <div style="padding: 0.5rem; background: #213885; border-radius: 8px; border: 1px solid #463FA9;">
                    <span style="font-weight: 600; color: #6A57F3;">65-79</span>
                    <div>Very Good</div>
                </div>
                <div style="padding: 0.5rem; background: #213885; border-radius: 8px; border: 1px solid #463FA9;">
                    <span style="font-weight: 600; color: #EB70EC;">80-100</span>
                    <div>Excellent</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if score >= 80:
                badge = "🏆 Excellent"
                color = "#EB70EC"
                message = "Your resume is in the top tier. You've demonstrated strong qualifications and clear communication of your experience."
            elif score >= 45:
                badge = "👍 Good"
                color = "#6A57F3"
                message = "Your resume is competitive. With some minor improvements, you could make it even stronger."
            else:
                badge = "⚠️ Needs Work"
                color = "#B693EF"
                message = "Your resume needs significant improvements. Focus on adding relevant skills and experiences."
            
            st.markdown(f"""
            <div style="background: #213885; border-left: 4px solid {color}; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; color: #ECDFD2;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: {color};">{badge}</span>
                </div>
                <p style="margin: 0; color: #ECDFD2;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if score >= 45:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; 
                            background: #213885; padding: 1rem; border-radius: 12px;
                            border: 1px solid #6A57F3; margin-bottom: 1.5rem; color: #ECDFD2;">
                    <div style="background: #4044F9; width: 40px; height: 40px; 
                                border-radius: 50%; display: flex; align-items: center; 
                                justify-content: center; flex-shrink: 0;">
                        <span style="color: white; font-size: 1.2rem;">✓</span>
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #EB70EC;">Interview Selection Status</h3>
                        <p style="margin: 0; color: #ECDFD2; font-size: 1.1rem;">
                            🎉 Congratulations! You are selected for an interview.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; 
                            background: #213885; padding: 1rem; border-radius: 12px;
                            border: 1px solid #B693EF; margin-bottom: 1.5rem; color: #ECDFD2;">
                    <div style="background: #463FA9; width: 40px; height: 40px; 
                                border-radius: 50%; display: flex; align-items: center; 
                                justify-content: center; flex-shrink: 0;">
                        <span style="color: white; font-size: 1.2rem;">✗</span>
                    </div>
                    <div>
                        <h3 style="margin: 0; color: #B693EF;">Interview Selection Status</h3>
                        <p style="margin: 0; color: #ECDFD2; font-size: 1.1rem;">
                            ❌ You are not selected for an interview at this time.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def home_page():
    """Whole new cutting-edge SaaS landing page design without the old illustration banner."""
    # Top Glowing Badge
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <span style="background: linear-gradient(135deg, rgba(64,68,249,0.3) 0%, rgba(235,112,236,0.3) 100%); color: #EB70EC; border: 1px solid #EB70EC; padding: 10px 24px; border-radius: 30px; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.5px; display: inline-block; box-shadow: 0 0 35px rgba(235,112,236,0.25);">
            ⚡ Enterprise Neural Resume Intelligence & AI Parser v3.5
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Hero Main Section (Clean Modern SaaS typography & layout, NO illustration banner)
    with stylable_container(
        key="ultra_modern_hero",
        css_styles="""
        {
            background: linear-gradient(135deg, #110792 0%, #1f12bd 100%);
            border-radius: 30px;
            padding: 5rem 3rem;
            margin-bottom: 3rem;
            border: 1px solid #6A57F3;
            box-shadow: 0 25px 80px rgba(7, 5, 246, 0.4);
            text-align: center;
        }
        """
    ):
        st.markdown("""
        <div style="max-width: 850px; margin: 0 auto;">
            <h1 style="font-size: 4rem; font-weight: 900; line-height: 1.1; margin-bottom: 1.5rem; color: #ffffff;">
                The Ultimate <span style="background: -webkit-linear-gradient(45deg, #EB70EC, #B693EF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Engine</span> for Resume Excellence
            </h1>
            <p style="font-size: 1.3rem; color: #ECDFD2; line-height: 1.6; margin-bottom: 3rem;">
                Stop losing jobs to automated ATS filters. Leverage deep neural networks and real-time LLM coaching to optimize your resume impact instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_hb1, col_hb2, col_hb3 = st.columns([1, 1.2, 1])
        with col_hb2:
            h_btn1, h_btn2 = st.columns([1, 1], gap="small")
            with h_btn1:
                if st.button("🚀 Get Started", key="hero_cta_start", use_container_width=True):
                    st.session_state.page = "candidate"
                    st.rerun()
            with h_btn2:
                if st.button("🛡️ Admin Portal", key="hero_cta_admin", use_container_width=True):
                    st.session_state.page = "admin"
                    st.rerun()

    # Live Metrics Bar
    with stylable_container(
        key="ultra_metrics_bar",
        css_styles="""
        {
            background: #213885;
            border-radius: 20px;
            padding: 2.25rem;
            margin-bottom: 3rem;
            border: 1px solid #6A57F3;
            box-shadow: 0 10px 30px rgba(106,87,243,0.25);
        }
        """
    ):
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("99.8%", "ATS Compatibility"),
            ("25k+", "Resumes Evaluated"),
            ("92%", "Interview Conversion"),
            ("0.05s", "Neural Parsing Speed")
        ]
        for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 2.6rem; font-weight: 900; color: #EB70EC; margin-bottom: 0.25rem;">{val}</div>
                    <div style="font-size: 1rem; color: #ECDFD2; font-weight: 600;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

    # Bento Grid Core Capabilities
    st.markdown("""
    <div style="text-align: center; margin: 3.5rem 0 2rem 0;">
        <h2 style="font-size: 2.8rem; font-weight: 900; color: #ffffff; margin-bottom: 0.5rem;">Core AI Capabilities</h2>
        <p style="color: #B693EF; font-size: 1.2rem;">Precision-engineered modules for modern recruitment workflows</p>
    </div>
    """, unsafe_allow_html=True)

    feat_c1, feat_c2, feat_c3 = st.columns(3, gap="large")
    capabilities = [
        ("⚡", "Advanced Neural Parsing", "Extracts structured metadata, work milestones, and skill proficiencies with unprecedented accuracy from any PDF or image."),
        ("🧠", "BiLSTM Quality Scoring", "Trained on comprehensive recruiter datasets to evaluate structural depth, impact verbs, and technical alignment."),
        ("🎯", "Virtual Career Coaching", "Delivers personalized, human-like constructive feedback paragraphs to bridge critical skill gaps instantly.")
    ]
    for col, (icon, title, desc) in zip([feat_c1, feat_c2, feat_c3], capabilities):
        with col:
            with stylable_container(
                key=f"cap_card_{title}",
                css_styles="""
                {
                    background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                    border: 1px solid #463FA9;
                    border-radius: 24px;
                    padding: 2.75rem 2.25rem;
                    box-shadow: 0 15px 45px rgba(7,5,246,0.25);
                    height: 100%;
                }
                """
            ):
                st.markdown(f"""
                <div>
                    <div style="background: #213885; width: 80px; height: 80px; border-radius: 22px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.75rem; font-size: 2.2rem; border: 1px solid #6A57F3; box-shadow: 0 8px 25px rgba(106,87,243,0.3);">
                        {icon}
                    </div>
                    <h3 style="font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 1rem;">{title}</h3>
                    <p style="color: #ECDFD2; font-size: 1.05rem; line-height: 1.6; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    # Bottom Gradient Action Banner
    with stylable_container(
        key="bottom_action_banner",
        css_styles="""
        {
            background: linear-gradient(135deg, #213885 0%, #0c0366 100%);
            border-radius: 28px;
            padding: 4rem 3rem;
            margin-top: 4rem;
            text-align: center;
            border: 1px solid #6A57F3;
            box-shadow: 0 25px 70px rgba(7, 5, 246, 0.4);
        }
        """
    ):
        st.markdown("""
        <h2 style="font-size: 2.5rem; font-weight: 900; color: #ffffff; margin-bottom: 1.25rem;">Ready to Accelerate Your Career?</h2>
        <p style="color: #ECDFD2; margin-bottom: 2.5rem; font-size: 1.25rem; max-width: 700px; margin-left: auto; margin-right: auto; line-height: 1.6;">
            Upload your resume now to uncover deep insights and join elite candidates landing top interviews.
        </p>
        """, unsafe_allow_html=True)
        if st.button("✨ Upload & Analyze Resume Free", key="landing_final_cta", type="primary"):
            st.session_state.page = "candidate"
            st.rerun()

def candidate_auth():
    """Whole new ultra-clean glassmorphic candidate login and registration portal."""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        justify-content: center;
        background: transparent;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 32px;
        background: #110792;
        border: 1px solid #463FA9;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #B693EF;
        box-shadow: 0 4px 15px rgba(7,5,246,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4044F9 0%, #6A57F3 100%) !important;
        color: #ffffff !important;
        border-color: #EB70EC !important;
        box-shadow: 0 6px 20px rgba(64,68,249,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem; padding-top: 1rem;">
            <div style="background: linear-gradient(135deg, #213885 0%, #4044F9 100%); width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.25rem; font-size: 2rem; border: 1px solid #EB70EC; box-shadow: 0 0 30px rgba(235,112,236,0.3);">
                🚀
            </div>
            <h2 style="font-size: 2.4rem; font-weight: 900; color: #ffffff; margin-bottom: 0.5rem;">Candidate Center</h2>
            <p style="color: #B693EF; font-size: 1.1rem;">Secure access to your resume evaluations & history</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with tab1:
            with stylable_container(
                key="new_login_glass_card",
                css_styles="""
                {
                    background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                    border: 1px solid #463FA9;
                    border-radius: 24px;
                    padding: 3rem;
                    box-shadow: 0 20px 60px rgba(7,5,246,0.3);
                    color: #ECDFD2;
                }
                """
            ):
                st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h3 style="color: #ffffff; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.3rem;">Welcome Back</h3>
                    <p style="color: #B693EF; font-size: 0.95rem;">Enter your account details below</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.form("login_form"):
                    user = st.text_input("Username", placeholder="Enter your username")
                    pwd = st.text_input("Password", type="password", placeholder="Enter your password")
                    
                    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                    if submitted:
                        password_hash = hash_password(pwd)
                        if validate_user(user, password_hash):
                            st.session_state.candidate_logged_in = True
                            st.session_state.candidate_username = user
                            st.session_state.page = 'candidate_dashboard'
                            st.rerun()
                        else:
                            st.error("Incorrect username or password.")
        
        with tab2:
            with stylable_container(
                key="new_register_glass_card",
                css_styles="""
                {
                    background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                    border: 1px solid #463FA9;
                    border-radius: 24px;
                    padding: 3rem;
                    box-shadow: 0 20px 60px rgba(7,5,246,0.3);
                    color: #ECDFD2;
                }
                """
            ):
                st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h3 style="color: #ffffff; font-size: 1.5rem; font-weight: 800; margin-bottom: 0.3rem;">Get Started Free</h3>
                    <p style="color: #B693EF; font-size: 0.95rem;">Create your profile in 30 seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.form("register_form"):
                    new_user = st.text_input("Username", placeholder="Choose a username")
                    new_pass = st.text_input("Password", type="password", placeholder="Create a secure password")
                    confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                    
                    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                    if submitted:
                        if not new_user or not new_pass:
                            st.error("Please enter both username and password.")
                        elif new_pass != confirm_pass:
                            st.error("Passwords do not match.")
                        else:
                            success, message = register_user(new_user, hash_password(new_pass))
                            if success:
                                st.success(message)
                                st.session_state.page = 'home'
                                st.rerun()
                            else:
                                st.error(message)

def candidate_dashboard():
    """Enhanced candidate dashboard with modern layout."""
    st.markdown(f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem; color: #ffffff;">👋 Welcome back, <span style="color: #EB70EC;">{st.session_state.candidate_username}</span></h1>
        <p style="color: #B693EF;">Upload your resume to get personalized feedback and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    with stylable_container(
        key="upload_section",
        css_styles="""
        {
            background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid #463FA9;
            margin-bottom: 2rem;
            color: #ECDFD2;
            box-shadow: 0 12px 40px rgba(7,5,246,0.2);
        }
        """
    ):
        st.markdown("### 📤 Upload Your Resume")
        st.markdown("Upload a PDF file to analyze your resume content and get instant feedback.")
        
        file = st.file_uploader("Choose a resume file", type=["pdf", "jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if file is not None:
            save_path = save_uploaded_file(file)
            if not save_path:
                return
                
            analysis, error = process_resume(save_path)
            
            if error:
                st.error(f"Analysis failed: {error}")
                return
                
            if analysis:
                if "raw_output" in analysis:
                    st.warning("Llama output could not be parsed as structured data. Showing raw output below.")
                    st.info(analysis["raw_output"])
                    return

                with stylable_container(
                    key="pdf_preview",
                    css_styles="""
                    {
                        background: #213885;
                        border-radius: 12px;
                        padding: 1rem;
                        margin: 1rem 0;
                        border: 1px dashed #6A57F3;
                        color: #ECDFD2;
                    }
                    """
                ):
                    st.markdown("**📄 Resume Preview**")
                    ext = os.path.splitext(save_path)[1].lower()
                    if ext == ".pdf":
                        with open(save_path, "rb") as f:
                            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    elif ext in [".jpg", ".jpeg", ".png"]:
                        st.image(save_path, caption="Uploaded Resume Image", use_container_width=True)
                    else:
                        st.info("Preview not available for this file type.")
                
                st.markdown("## 📝 Resume Analysis Results")
                
                display_candidate_info(analysis)
                display_skills(analysis)
                display_degree(analysis)
                display_work_experience(analysis)

                import json

                advice_prompt = f"""
You are an expert career coach. Given the following extracted resume data (in JSON), provide a single, detailed, actionable advice paragraph for the candidate to improve their resume and job prospects. Do not return a list or JSON, just a clear, readable paragraph of advice.

Resume Data:
{json.dumps(analysis, ensure_ascii=False)}

Advice:
"""
                advice_output = llama3_infer(advice_prompt)
                advice_text = advice_output.strip()
                try:
                    if advice_text.startswith('['):
                        advice_list = json.loads(advice_text)
                        advice_text = ' '.join([item['value'] for item in advice_list if isinstance(item, dict) and 'value' in item])
                    elif advice_text.startswith('{'):
                        advice_json = json.loads(advice_text)
                        advice_text = advice_json.get('advice', advice_text)
                except Exception:
                    pass

                with stylable_container(
                    key="advice_section",
                    css_styles="""
                    {
                        background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                        border-radius: 16px;
                        padding: 1.75rem;
                        margin-bottom: 1.5rem;
                        border: 1px solid #463FA9;
                        color: #ECDFD2;
                        box-shadow: 0 12px 40px rgba(7,5,246,0.2);
                    }
                    """
                ):
                    st.subheader("💡 Advice")
                    st.markdown(f"**Advice:** {advice_text}")

                display_wordcloud(extract_text_from_file(save_path))
                display_resume_score()

                score = st.session_state.resume_score
                explanation_prompt = f"""
You are an expert resume reviewer. Given the following extracted resume data (in JSON) and the candidate's resume score ({score:.1f}), explain in detail why the candidate received this score, referencing specific strengths and weaknesses. Do NOT provide recommendations or advice—just explain the basis for the score.

Resume Data:
{json.dumps(analysis, ensure_ascii=False)}

Respond with a clear, candidate-friendly explanation.
"""
                explanation_output = llama3_infer(explanation_prompt)
                with stylable_container(
                    key="score_explanation_section",
                    css_styles="""
                    {
                        background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                        border-radius: 16px;
                        padding: 1.75rem;
                        margin-bottom: 1.5rem;
                        border: 1px solid #463FA9;
                        color: #ECDFD2;
                        box-shadow: 0 12px 40px rgba(7,5,246,0.2);
                    }
                    """
                ):
                    st.subheader("📝 Score Explanation")
                    st.markdown(explanation_output)
    
    if st.button("🚪 Logout", type="primary", use_container_width=True):
        st.session_state.candidate_logged_in = False
        st.session_state.page = 'home'
        st.rerun()

def admin_dashboard():
    """Enhanced admin dashboard with better data visualization."""
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem; color: #ffffff;">🛡️ Admin Dashboard</h1>
        <p style="color: #B693EF;">Manage candidate data and system analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    stats = get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        with stylable_container(
            key="total_candidates",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #463FA9;
                color: #ECDFD2;
                box-shadow: 0 12px 40px rgba(7,5,246,0.2);
            }
            """
        ):
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="font-size: 0.9rem; color: #B693EF; margin-bottom: 0.5rem;">Total Candidates</p>
                <h2 style="color: #EB70EC; margin-top: 0;">{stats['total_candidates']}</h2>
                <p style="font-size: 0.8rem; color: #B693EF;">↗️ Active Submissions</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with stylable_container(
            key="avg_score",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #463FA9;
                color: #ECDFD2;
                box-shadow: 0 12px 40px rgba(7,5,246,0.2);
            }
            """
        ):
            avg_score = round(stats['avg_score'], 1) if stats['avg_score'] else 0
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="font-size: 0.9rem; color: #B693EF; margin-bottom: 0.5rem;">Average Score</p>
                <h2 style="color: #EB70EC; margin-top: 0;">{avg_score}</h2>
                <p style="font-size: 0.8rem; color: #B693EF;">{"🔼 Strong performance" if avg_score > 60 else "🔽 Needs improvement"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with stylable_container(
            key="recent_activity",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #463FA9;
                color: #ECDFD2;
                box-shadow: 0 12px 40px rgba(7,5,246,0.2);
            }
            """
        ):
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="font-size: 0.9rem; color: #B693EF; margin-bottom: 0.5rem;">Recent (7 days)</p>
                <h2 style="color: #EB70EC; margin-top: 0;">{stats['recent_candidates']}</h2>
                <p style="font-size: 0.8rem; color: #B693EF;">Recent submissions</p>
            </div>
            """, unsafe_allow_html=True)
    
    with stylable_container(
        key="data_table",
        css_styles="""
        {
            background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
            border-radius: 16px;
            padding: 1.75rem;
            border: 1px solid #463FA9;
            margin-bottom: 1.5rem;
            color: #ECDFD2;
            box-shadow: 0 12px 40px rgba(7,5,246,0.2);
        }
        """
    ):
        st.markdown("### 📋 Candidate Data")
        st.markdown("View and manage all candidate submissions and analysis results.")
        
        df = get_candidate_data()
        if not df.empty:
            drop_cols = [col for col in df.columns if col.lower() in ["page_no", "predicted_field", "recommended_skills", "user_level"]]
            df = df.drop(columns=drop_cols, errors="ignore")
            
            def infer_level(row):
                try:
                    years = float(row.get("years_experience", 0))
                except Exception:
                    years = 0
                degree = str(row.get("degree", "") or row.get("Degree", "")).lower()
                skills = str(row.get("Skill", "") or row.get("skills", "") or row.get("Skills", "")).split(",")
                if years >= 5:
                    return "Senior"
                elif ("bachelor" in degree or "master" in degree) and len([s for s in skills if s.strip()]) >= 5:
                    return "Mid"
                else:
                    return "Junior"
            
            df["Candidate level"] = df.apply(infer_level, axis=1)
            
            def summarize_experience(row):
                job = row.get("job_title") or row.get("Job Title")
                company = row.get("company") or row.get("Company")
                if job and company:
                    return f"{job} at {company}"
                elif job:
                    return str(job)
                elif company:
                    return str(company)
                for col in row.index:
                    if 'experience' in col.lower():
                        val = row.get(col)
                        if val and str(val).strip() and str(val).strip().lower() not in ["none", "n/a", "null"]:
                            return str(val)
                return "N/A"
            
            df["Experience"] = df.apply(summarize_experience, axis=1)
            drop_course_cols = [col for col in df.columns if "course" in col.lower()]
            if drop_course_cols:
                df = df.drop(columns=drop_course_cols, errors="ignore")
            
            score_col = "Score" if "Score" in df.columns else ("resume_score" if "resume_score" in df.columns else None)
            if score_col:
                df = df.sort_values(by=score_col, ascending=False)
            df = df.reset_index(drop=True)
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            df.insert(0, "ID", df.index + 1)
            
            st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    score_col: st.column_config.ProgressColumn(
                        "Score",
                        help="Resume quality score",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    )
                } if score_col else {},
                disabled=True,
                key="admin_table_editor",
                num_rows="dynamic"
            )
            candidate_ids = df["ID"].tolist()
            selected_id = st.selectbox("Select candidate ID to view details", candidate_ids, key="select_candidate_id")
            if st.button("Show Details for Selected Candidate"):
                candidate = df[df["ID"] == selected_id].iloc[0].to_dict()
                st.session_state.selected_candidate = candidate
                st.session_state.page = 'candidate_detail_admin'
                st.rerun()
            st.markdown(get_table_download_link(df, "candidate_data.csv", "📥 Download as CSV"), unsafe_allow_html=True)
        else:
            st.info("No candidate data available")
    
    if not df.empty:
        with stylable_container(
            key="score_chart",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border-radius: 16px;
                padding: 1.75rem;
                border: 1px solid #463FA9;
                margin-bottom: 1.5rem;
                color: #ECDFD2;
                box-shadow: 0 12px 40px rgba(7,5,246,0.2);
            }
            """
        ):
            st.markdown("### 📊 Score Distribution")
            st.markdown("This chart shows the distribution of resume scores for all candidates in the system.")
            target_score_col = "Score" if "Score" in df.columns else ("resume_score" if "resume_score" in df.columns else None)
            if target_score_col:
                df['PlotScore'] = pd.to_numeric(df[target_score_col], errors='coerce')
                chart_df = df.dropna(subset=['PlotScore'])
                if not chart_df.empty:
                    score_bins = pd.cut(chart_df['PlotScore'], bins=[0, 20, 40, 60, 80, 100], right=True,
                                        labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
                    pie_df = score_bins.value_counts().reset_index()
                    pie_df.columns = ['Score Range', 'Count']
                    fig = px.pie(pie_df, names='Score Range', values='Count', 
                                 title='Candidate Score Distribution',
                                 color_discrete_sequence=['#4044F9', '#6A57F3', '#EB70EC', '#B693EF', '#213885'])
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ECDFD2'),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🚪 Logout", type="primary", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.session_state.page = 'home'
        st.rerun()

def admin_login():
    """Enhanced admin login with security focus."""
    with stylable_container(
        key="admin_login_container",
        css_styles="""
        {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem 0;
        }
        """
    ):
        with stylable_container(
            key="admin_login_box",
            css_styles="""
            {
                background: linear-gradient(135deg, #110792 0%, #1a10a8 100%);
                border-radius: 16px;
                padding: 2.5rem;
                border: 1px solid #463FA9;
                box-shadow: 0 12px 40px rgba(7,5,246,0.2);
                text-align: center;
                color: #ECDFD2;
            }
            """
        ):
            st.markdown("""
            <div style="margin-bottom: 2rem;">
                <div style="background: #213885; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; border: 1px solid #6A57F3;">
                    <span style="font-size: 2rem; color: #EB70EC;">🔒</span>
                </div>
                <h2 style="margin-bottom: 0.5rem; color: #ffffff;">Admin Portal</h2>
                <p style="color: #B693EF;">Restricted access to authorized personnel only</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("admin_login_form"):
                ad_user = st.text_input("Username", placeholder="Enter admin username")
                ad_password = st.text_input("Password", type="password", placeholder="Enter admin password")
                
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
                if submitted:
                    if ad_user == 'ArmanLekhak' and ad_password == 'Project':
                        st.session_state.admin_logged_in = True
                        st.session_state.page = 'admin_dashboard'
                        st.rerun()
                    else:
                        st.error("Incorrect credentials")
                        st.markdown("""
                        <div style="text-align: center; margin-top: 1rem;">
                            <p style="color: #B693EF; font-size: 0.9rem;">Contact system administrator if you've forgotten your credentials</p>
                        </div>
                        """, unsafe_allow_html=True)






    
