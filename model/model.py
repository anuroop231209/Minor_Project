import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import requests
import json
import logging
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Llama 3 Extraction (via LM Studio OpenAI API) ---
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

#Enhanced Prompt Templates
LLAMA3_EXTRACTION_PROMPT_TEMPLATES = [
        """
    You are an expert resume parser. The following text is extracted from a scanned or image-based resume using OCR, so it may contain noise, symbols, layout artifacts, or repeated/misplaced fields. Extract the following fields from the resume text below and respond ONLY with valid JSON (no explanations, no free text, no markdown, no comments, no code blocks, no extra text, no triple backticks, no labels, no headings, no preamble, no postamble, no formatting, no extra whitespace). Output must be a single valid JSON object and nothing else.

    Special instructions for noisy OCR text:
    - Ignore all symbols, layout artifacts, and repeated or misplaced fields.
    - If fields appear multiple times, use the most complete or most recent value.
    - Extract fields even if they are separated by noise or appear out of order.
    - Focus on extracting:
    - Name, email, phone, skills, years_experience, degree, universities, summary, experiences (job_title, company, start_date, end_date, responsibilities)
    - Ignore headers, footers, page numbers, and irrelevant sections.
    - If the resume is a scan or has poor formatting, do your best to extract the main fields, but do not hallucinate or invent data.

    Required JSON structure:
    {
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "years_experience": "",
        "degree": [],
        "universities": [],
        "summary": "",
        "experiences": [
            {
                "job_title": "",
                "company": "",
                "start_date": "",
                "end_date": "",
                "responsibilities": ""
            }
        ]
    }

    Example of good output:
    {
        "name": "Jane Doe",
        "email": "jane.doe@email.com",
        "phone": "+1234567890",
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "years_experience": "5",
        "degree": ["BSc Computer Science"],
        "universities": ["University of Example"],
        "summary": "Experienced data scientist with a strong background in machine learning and analytics.",
        "experiences": [
            {
                "job_title": "Data Scientist",
                "company": "TechCorp",
                "start_date": "2018-06",
                "end_date": "2023-01",
                "responsibilities": "Developed predictive models and performed data analysis."
            }
        ]
    }

    Instructions:
    - Output ONLY valid JSON, no extra text, no markdown, no code blocks, no explanations, no comments, no headings, no preamble, no postamble, no formatting, no extra whitespace.
    - If a field is missing, use "" or [] (do not invent or hallucinate).
    - For skills and degree, always output a list of strings.
    - For experiences, always output a list of objects with all keys present.
    - Be precise with dates (use YYYY-MM format when possible).
    - Extract responsibilities as a single string or list of bullet points.

     Resume Text:
    """,
    """
    You are an advanced resume analysis AI. Carefully parse the following resume text and extract structured information in JSON format.
    
        Required JSON schema:
    {
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+1234567890",
        "skills": ["Skill1", "Skill2"],
        "years_experience": "5",
        "degree": ["Bachelor of Science in Computer Science"],
        "universities": ["University Name"],
        "summary": "Professional summary text",
        "experiences": [
            {
                "job_title": "Position Title",
                "company": "Company Name",
                "start_date": "2020-01",
                "end_date": "2023-12",
                "responsibilities": "Job responsibilities description"
            }
        ]
    }

    Guidelines:
    - Prioritize accuracy over completeness
    - Handle missing fields gracefully (use empty values)
    - Normalize dates to YYYY-MM format when possible
    - Extract skills as a list of specific technical/professional skills
    - For experiences, capture at least the most recent 3 positions
    
    Resume Text:
    """,
    """
    [SYSTEM] You are a top-tier resume parsing system. Extract the maximum possible information from the resume below, following the JSON structure exactly.
    [USER] Resume Text:
    """
]

def llama_infer(prompt: str, n_predict: int =1024, temperature: float=0.2)->str:
    headers = {"Content-Type": "application/json"}
    data = {
        "model": LMSTUDIO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": temperature,
        "stop": ["</s>", "```"]
    }
    try:
        start_time = time.time()
        response = requests.post(LMSTUDIO_API_URL, headers=headers, json=data, timeout=180)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        elapsed = time.time() - start_time
        logging.info(f"Llama inference completed in {elapsed:.2f}s, tokens: {len(content.split())}")
        return content
    except Exception as e:
        logging.error(f"LM Studio API call failed: {e}")
        return f"[ERROR] LM Studio API call failed: {e}"

def extract_first_json(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from text with enhanced parsing."""
    # Try to find JSON within triple backticks
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()
    
    # Find the first complete JSON object
    json_start = text.find('{')
    if json_start == -1:
        return None
        
    brace_count = 0
    in_string = False
    escape = False
    json_str = ""
    
    for char in text[json_start:]:
        if char == '{' and not in_string:
            brace_count += 1
        elif char == '}' and not in_string:
            brace_count -= 1
        elif char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
        else:
            escape = False
            
        json_str += char
        
        if brace_count == 0 and not in_string:
            break
    
    # Try to parse the JSON
    try:
        # Clean up common issues
        json_str = json_str.replace('\n', ' ').replace('\t', ' ')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parsing failed: {e}")
        return None

def validate_resume_json(data: Dict) -> bool:
    """Validate the structure and content of extracted resume data."""
    if not isinstance(data, dict):
        return False
        
    # Check required fields
    required = ['name', 'email', 'skills', 'experiences']
    if any(field not in data for field in required):
        return False
        
    # Check data types
    if not isinstance(data.get('skills', []), list):
        return False
    if not isinstance(data.get('experiences', []), list):
        return False
        
    # Check name field
    if not data.get('name') or not isinstance(data['name'], str) or len(data['name'].strip()) < 2:
        return False
        
    # Check email format
    email = data.get('email', '')
    if isinstance(email, list):
        # If email is a list, try to extract the first string
        email = email[0] if email and isinstance(email[0], str) else ''
    if email and isinstance(email, str) and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False
        
    # Check experiences structure
    for exp in data.get('experiences', []):
        if not isinstance(exp, dict):
            return False
        if 'job_title' not in exp or 'company' not in exp:
            return False
            
    return True
    
def parse_experiences(experiences: List[Dict]) -> List[Dict]:
    """Normalize and clean experiences data."""
    parsed = []
    for exp in experiences:
        # Normalize dates
        start_date = exp.get('start_date', '')
        end_date = exp.get('end_date', 'Present')
        
        # Extract year from dates
        if start_date:
            year_match = re.search(r'\d{4}', start_date)
            if year_match:
                start_date = year_match.group(0)
                
        if end_date and end_date != 'Present':
            year_match = re.search(r'\d{4}', end_date)
            if year_match:
                end_date = year_match.group(0)
                
        # Clean responsibilities
        responsibilities = exp.get('responsibilities', '')
        if isinstance(responsibilities, list):
            responsibilities = ' '.join(responsibilities)
            
        parsed.append({
            'job_title': exp.get('job_title', ''),
            'company': exp.get('company', ''),
            'start_date': start_date,
            'end_date': end_date,
            'responsibilities': responsibilities[:500]  # Limit length
        })
        
    return parsed
