import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


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

