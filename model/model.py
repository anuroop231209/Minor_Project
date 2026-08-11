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

def extract_individual_fields(resume_text: str) -> Dict[str, Any]:
    """Fallback method: extract each field individually with specialized prompts."""
    fields = {
        "name": "Extract the candidate's full name from the resume text. Respond ONLY with the name itself, no explanations, no extra text, no context, no labels, no markdown, no comments. If not found, respond with N/A only.",
        "email": "Extract the candidate's email address from the resume text. Respond ONLY with the email itself, no explanations, no extra text, no context, no labels, no markdown, no comments. If not found, respond with N/A only.",
        "phone": "Extract the candidate's phone number from the resume text. Respond ONLY with the phone number itself, no explanations, no extra text, no context, no labels, no markdown, no comments. If not found, respond with N/A only.",
        "skills": "Extract the candidate's skills as a JSON list from the resume text. Respond ONLY with a JSON list of strings, no explanations, no extra text, no context, no labels, no markdown, no comments.",
        "summary": "Extract a professional summary from the resume text. Respond ONLY with a string of 2-3 sentences, no explanations, no extra text, no context, no labels, no markdown, no comments.",
    }
    
    result = {}
    for field, prompt in fields.items():
        full_prompt = f"{prompt}\n\nResume Text:\n{resume_text}\n\nOutput:"
        response = llama3_infer(full_prompt, temperature=0.1)
        
        # Clean response
        response = response.strip().replace('"', '').replace("'", "")
        
        # Special handling for skills
        if field == "skills":
            try:
                if response.startswith('[') and response.endswith(']'):
                    skills = json.loads(response)
                else:
                    skills = [s.strip() for s in response.split(',') if s.strip()]
                result[field] = skills
            except:
                result[field] = []
        else:
            result[field] = response if response and response != 'N/A' else ""
    
    # Set default values for missing complex fields
    result['degree'] = []
    result['universities'] = []
    result['years_experience'] = ""
    result['experiences'] = []

    # --- Regex-based fallback for critical fields ---
    # Email
    if not result.get('email') or result['email'].strip().lower() in ["n/a", "", "none", "null"]:
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", resume_text)
        if email_match:
            result['email'] = email_match.group(0)
        else:
            result['email'] = "N/A"
    # Phone
    if not result.get('phone') or result['phone'].strip().lower() in ["n/a", "", "none", "null"]:
        phone_match = re.search(r"(\+?\d[\d\s\-()]{7,}\d)", resume_text)
        if phone_match:
            result['phone'] = phone_match.group(0)
        else:
            result['phone'] = "N/A"
    # Name (very basic fallback: first capitalized words at top)
    if not result.get('name') or result['name'].strip().lower() in ["n/a", "", "none", "null"]:
        # Try to find a likely name in the first 10 lines
        lines = resume_text.splitlines()
        for line in lines[:10]:
            line = line.strip()
            if len(line.split()) >= 2 and all(w[0].isupper() for w in line.split() if w):
                result['name'] = line
                break
        if not result.get('name') or result['name'].strip() == "":
            result['name'] = "N/A"
    
    return result
def llama3_extract_resume_info(resume_text: str, attempt: int = 1, file_path: str = None) -> Dict[str, Any]:
    """
    Extract resume information using Llama 3 with enhanced techniques:
    1. Multiple prompt templates
    2. Progressive fallback strategies
    3. Validation and normalization
    4. Individual field extraction fallback
    5. Robust OCR pipeline for image-based PDFs and images
    """
    import mimetypes
    import re
    # Determine if file needs OCR
    needs_ocr = False
    if file_path is not None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            needs_ocr = True
        elif ext == '.pdf':
            # Heuristic: if text is very short or contains mostly non-ASCII, treat as image-based
            ascii_text = re.sub(r'[^\x00-\x7F]+', '', resume_text)
            if len(ascii_text.strip()) < 50:
                needs_ocr = True
    # If needs OCR, run robust OCR pipeline
    if needs_ocr and attempt == 1:
        try:
            from PIL import Image, ImageFilter
            import pytesseract
            import cv2
            import numpy as np
            if file_path.lower().endswith('.pdf'):
                from pdf2image import convert_from_path
                images = convert_from_path(file_path)
            else:
                images = [Image.open(file_path)]
            ocr_text = ""
            for img in images:
                # Upscale if small
                if min(img.size) < 1000:
                    scale = 2 if min(img.size) > 500 else 3
                    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
                img = img.convert('L')
                img_cv = np.array(img)
                img_cv = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                img_cv = cv2.fastNlMeansDenoising(img_cv, None, 30, 7, 21)
                img = Image.fromarray(img_cv)
                img = img.filter(ImageFilter.SHARPEN)
                texts = []
                for psm in [3, 4, 6, 11]:
                    config = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(img, lang='eng', config=config)
                    texts.append(text)
                all_lines = set()
                for t in texts:
                    for line in t.splitlines():
                        if line.strip():
                            line = line.replace(' @ ', '@').replace(' . ', '.').replace(' (at) ', '@').replace(' [at] ', '@')
                            line = line.replace(' dot ', '.').replace(' [dot] ', '.')
                            all_lines.add(line.strip())
                merged = "\n".join(sorted(all_lines))
                merged = re.sub(r'[^\x00-\x7F]+', ' ', merged)
                merged = re.sub(r'\s+', ' ', merged)
                ocr_text += merged + "\n"
            cleaned_text = ocr_text.strip()
            # Try again with cleaned OCR text
            return llama3_extract_resume_info(cleaned_text, attempt=2, file_path=file_path)
        except Exception as e:
            logging.error(f"OCR pipeline failed: {e}")
            # Fallback to original text
    # Select prompt based on attempt number
    prompt_idx = min(attempt - 1, len(LLAMA3_EXTRACTION_PROMPT_TEMPLATES) - 1)
    # Use a more detailed prompt for OCR cases
    if needs_ocr:
        detailed_prompt = """
You are an expert resume parser. The following text is extracted from a scanned or image-based resume using OCR, so it may contain errors, missing spaces, or formatting issues. Extract ALL possible fields with best effort, even if the text is messy. Use context and common resume patterns to infer missing fields. Respond ONLY with valid JSON (no explanations, no free text, no markdown, no comments, no code blocks, no extra text, no triple backticks, no labels, no headings, no preamble, no postamble, no formatting, no extra whitespace). Output must be a single valid JSON object and nothing else.

Focus on extracting:
- Name, email, phone, skills, years_experience, degree, universities, summary, experiences (job_title, company, start_date, end_date, responsibilities)
- If a field is missing, use "" or [] (do not invent or hallucinate, but do your best to infer from context)
- For skills and degree, always output a list of strings
- For experiences, always output a list of objects with all keys present
- Be precise with dates (use YYYY-MM format when possible)
- Extract responsibilities as a single string or list of bullet points
- If you see a section header (e.g. "Skills", "Experience", "Education"), treat the following lines as belonging to that section
- If the text is messy, use your best judgment to reconstruct the correct information

Resume Text:
"""
        prompt = detailed_prompt + resume_text
    else:
        prompt = LLAMA3_EXTRACTION_PROMPT_TEMPLATES[prompt_idx] + resume_text
    logging.info(f"Extraction attempt #{attempt} with prompt template {prompt_idx}")
    output = llama3_infer(prompt, n_predict=1024, temperature=0.3)
    parsed = extract_first_json(output)
    if parsed and validate_resume_json(parsed):
        parsed['experiences'] = parse_experiences(parsed.get('experiences', []))
        skills = parsed.get('skills', [])
        if isinstance(skills, str):
            parsed['skills'] = [s.strip() for s in skills.split(',') if s.strip()]
        elif isinstance(skills, list):
            parsed['skills'] = [str(s).strip() for s in skills]
        else:
            parsed['skills'] = []
        degree = parsed.get('degree', [])
        if isinstance(degree, str):
            parsed['degree'] = [degree.strip()]
        elif isinstance(degree, list):
            parsed['degree'] = [str(d).strip() for d in degree]
        else:
            parsed['degree'] = []
        parsed['name'] = parsed.get('name', 'N/A').strip().title()
        logging.info("Successfully extracted structured resume data")
        return parsed
    if attempt >= 2:
        logging.warning("Falling back to individual field extraction")
        parsed = extract_individual_fields(resume_text)
        parsed['experiences'] = []
        return parsed
    return {
        "name": "N/A",
        "email": "N/A",
        "phone": "N/A",
        "skills": [],
        "years_experience": "N/A",
        "degree": "N/A",
        "universities": [],
        "summary": "N/A",
        "experiences": [],
        "fallback": True
    }

# --- PyTorch BiLSTM Model for Resume Scoring ---
class ResumeDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.texts = (
            self.df['career_objective'].fillna('') + ' ' +
            self.df['skills'].fillna('') + ' ' +
            self.df['responsibilities'].fillna('')
        )
        self.scores = self.df['matched_score'].values.astype(np.float32)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in self.texts:
            for word in str(text).split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        tokens = [self.vocab.get(w, self.vocab['<UNK>']) for w in text.split()]
        if len(tokens) < self.max_len:
            tokens += [self.vocab['<PAD>']] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.scores[idx], dtype=torch.float32)

class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        out = self.fc(h)
        return out.squeeze(1)

def train_bilstm(csv_path, model_out_path="bilstm_resume_score.pth", epochs=20, batch_size=32, lr=1e-3):
    dataset = ResumeDataset(csv_path)
    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = BiLSTMRegressor(vocab_size=len(dataset.vocab))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / train_size
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)
        avg_val_loss = val_loss / val_size
        print(f"[Epoch {epoch+1}] Train MSE: {avg_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'vocab': dataset.vocab}, model_out_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    print(f"[INFO] BiLSTM model trained and saved to {model_out_path}")
    return model, dataset.vocab

def bilstm_score_resume(text, model_path="model/bilstm_resume_score.pth", max_len=128):
    if not os.path.exists(model_path):
        return None  # Model file missing
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab = checkpoint['vocab']
        model = BiLSTMRegressor(vocab_size=len(vocab))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tokens = [vocab.get(w, vocab['<UNK>']) for w in str(text).split()]
        if len(tokens) < max_len:
            tokens += [vocab['<PAD>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        X = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            score = model(X).item()
        return float(score)
    except Exception as e:
        return None  # Could not load or run model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Llama 3 Extraction and BiLSTM Resume Scoring")
    parser.add_argument("train_bilstm", nargs="?", help="Train BiLSTM on resume_data.csv", default=None)
    parser.add_argument("--csv_path", type=str, help="Path to resume_data.csv")
    parser.add_argument("--model_out_path", type=str, default="bilstm_resume_score.pth", help="Output path for BiLSTM model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.train_bilstm:
        train_bilstm(args.csv_path, args.model_out_path, args.epochs, args.batch_size, args.lr)
    else:
        print("""
Usage:
  python model.py train_bilstm --csv_path resume_data.csv [--model_out_path bilstm_resume_score.pth] [--epochs 10] [--batch_size 16] [--lr 0.001]

Description:
  train_bilstm - Train BiLSTM on your resume_data.csv for resume scoring
        """)