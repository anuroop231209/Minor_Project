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

        
    }


    """

]

