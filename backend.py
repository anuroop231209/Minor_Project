import os
import io
import base64
import hashlib
import datetime
import contextlib
import pymysql
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
from PyPDF2 import PdfReader
from typing import Optional, Tuple, Dict, Any, List

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'AnuroopTater',
    'password': 'Project',
    'db': 'candidate',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# --- Database Connection Utilities ---
@contextlib.contextmanager
def db_connection():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

@contextlib.contextmanager
def db_cursor():
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

# --- Database Setup ---
def setup_database():
    with db_cursor() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS candidate;")
        cursor.execute("USE candidate;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                ID INT NOT NULL AUTO_INCREMENT,
                Name VARCHAR(500) NULL,
                Email_ID VARCHAR(500) NULL,
                Score FLOAT NULL,
                Timestamp VARCHAR(50) NULL,
                `Candidate level` VARCHAR(50) NULL,
                Experience TEXT NULL,
                Skill TEXT NULL,
                PRIMARY KEY (ID)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL
            );
        """)

# Initialize database on import
setup_database()

# --- Security Utilities ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# --- File Hashing Utility ---
def hash_file(file_path: str) -> str:
    """Return the SHA256 hash of the file at file_path."""
    import hashlib
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

        
#PDF and Image Processing
def extract_text_from_pdf(file_path: str) -> str:
    try:
        # 1.Try PyPDF2
        from pyPDF2 import PdfReader
        reader = PdfReader(file_path)
        pypdf2_text="\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()
        if pypdf2_text and len(pypdf2_text)>30:
            return pypdf2_text
        
        #2. Try pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract_text
            pdfminer_text=pdfminer_extract_text(file_path)
            if pdfminer_text and len(pdfminer_text)>30:
                return pdfminer_text
        except Exception:
            pass

        # 3. Fallback: OCR only if both extractors fail
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(file_path)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang='eng') + "\n"
            ocr_text = ocr_text.strip()
            if ocr_text and len(ocr_text) > 30:
                return ocr_text
        except Exception:
            pass
        return ""
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")
    
from plt import Image, ImageEnhance, Imagefilter
import pytesseract
import re

def extract_text_from_file(file_path: str) -> str:
    import cv2
    import numpy as np
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        try:
            img = Image.open(file_path)
            # Upscale if small
            if min(img.size) < 1000:
                scale = 2 if min(img.size) > 500 else 3
                img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            # Grayscale
            img = img.convert('L')
            # Convert to OpenCV for advanced processing
            img_cv = np.array(img)
            # Adaptive thresholding
            img_cv = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # Denoising
            img_cv = cv2.fastNlMeansDenoising(img_cv, None, 30, 7, 21)
            # Convert back to PIL
            img = Image.fromarray(img_cv)
            # Sharpen
            img = img.filter(ImageFilter.SHARPEN)
            # Try all psm modes (0-13) and select the best result
            best_text = ""
            best_len = 0
            all_psm_texts = []
            for psm in range(0, 14):
                config = f'--oem 3 --psm {psm}'
                try:
                    text = pytesseract.image_to_string(img, lang='eng', config=config)
                except Exception:
                    text = ""
                all_psm_texts.append((psm, text))
                if text and len(text) > best_len:
                    best_text = text
                    best_len = len(text)
            # Optionally, combine all unique lines from all PSMs
            all_lines = set()
            for _, t in all_psm_texts:
                for line in t.splitlines():
                    if line.strip():
                        # Clean common OCR email/phone errors
                        line = line.replace(' @ ', '@').replace(' . ', '.').replace(' (at) ', '@').replace(' [at] ', '@')
                        line = line.replace(' dot ', '.').replace(' [dot] ', '.')
                        all_lines.add(line.strip())
            # Merge lines and remove duplicates
            merged = "\n".join(sorted(all_lines))
            # Remove non-ASCII and excessive whitespace
            merged = re.sub(r'[^\x00-\x7F]+', ' ', merged)
            merged = re.sub(r'\s+', ' ', merged)
            # Clean up: filter out lines that are mostly symbols or too short, and deduplicate
            def is_valid_line(line):
                # Remove lines that are mostly symbols or too short
                if len(line.strip()) < 5:
                    return False
                # If more than 60% of chars are non-alphanumeric, skip
                alnum = sum(c.isalnum() for c in line)
                if len(line) == 0 or alnum / len(line) < 0.4:
                    return False
                return True
            filtered_lines = [line for line in sorted(all_lines) if is_valid_line(line)]
            filtered_merged = "\n".join(filtered_lines)
            # Use the filtered merged text if it's longer, otherwise use the best single PSM
            if len(filtered_merged) > best_len:
                return filtered_merged.strip()
            else:
                return best_text.strip()
        except Exception as e:
            raise RuntimeError(f"Image extraction failed: {str(e)}")
    else:
        raise RuntimeError("Unsupported file type")

# --- Data Export Utilities ---
def get_table_download_link(df: pd.DataFrame, filename: str, text: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

# --- Database Operations ---
def insert_resume_data(
    name: str,
    email: str,
    score: float,
    timestamp: str,
    candidate_level: str,
    Skill: str,
    Experience: str,
) -> bool:
    try:
        with db_cursor() as cursor:
            required_fields = [name, email, score, timestamp, candidate_level, Skill, Experience]
            if any(f is None or (isinstance(f, str) and not f.strip()) for f in required_fields):
                print(f"[ERROR] One or more required fields are missing or empty. Data: {required_fields}")
                return False

            cursor.execute("""
                INSERT INTO user_data (
                    Name, Email_ID, Score, Timestamp,
                    `Candidate level`, Experience, Skill
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    name, email, score, timestamp,
                    candidate_level, Experience, Skill
                ))
            print(f"[INFO] Successfully inserted resume for {name} ({email}).")
        return True
    except Exception as e:
        print(f"[ERROR] Database insertion failed: {str(e)} | Data: name={name}, email={email}, score={score}")
        return False
    
def validate_user(username: str, password_hash: str) -> bool:
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM candidate_users WHERE username=%s AND password_hash=%s",
                (username, password_hash)
            )
            return cursor.fetchone() is not None
    except Exception as e:
        print(f"[ERROR] User validation failed: {str(e)}")
        return False

def register_user(username: str, password_hash: str) -> Tuple[bool, str]:
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "INSERT INTO candidate_users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash)
            )
        return True, "Registration successful"
    except pymysql.err.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"
    
def get_candidate_data() -> pd.DataFrame:
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT * FROM user_data")
            rows = cursor.fetchall()
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"[ERROR] Failed to fetch candidate data: {str(e)}")
        return pd.DataFrame()
    
def get_stats() -> Dict[str, Any]:
    stats = {}
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM user_data")
            stats['total_candidates'] = cursor.fetchone()['total']

            cursor.execute("SELECT AVG(resume_score) as avg_score FROM user_data")
            stats['avg_score'] = cursor.fetchone()['avg_score'] or 0

            cursor.execute("SELECT COUNT(*) as recent FROM user_data WHERE Timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)")
            stats['recent_candidates'] = cursor.fetchone()['recent']

        return stats
    except Exception as e:
        print(f"[ERROR] Failed to fetch stats: {str(e)}")
        return {'total_candidates': 0, 'avg_score': 0, 'recent_candidates': 0}

# --- Compatibility Stubs ---
def get_resume_score(*args, **kwargs):
    # Stub implementation to resolve import error
    return None
