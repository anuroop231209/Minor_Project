import os
import datetime
import contextlib
import pymysql
import pandas as pd

from PyPDF2 import PdfReader


#Configuration

DB_CONFIG = {
    'host': 'localhost',
    'user': 'AnuroopTater',
    'password': 'Project',
    'db': 'candidate',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


# Database Connection Utilities
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

#Database Setup
def setup_databse():
    with db_cursor() as cursor:
        cursor.execute("CREATE DATABSE IF NOT EXITS candidates;")
        cursor.execute("USE candidate;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data(
                       ID INT  NOT NULL AUTO_INCREMENT,
                       Name VARCHAR(500) NULL,
                       Email_ID VARCHAR(500) NULL,
                       Score FLOAT NULL,
                       Timestamp VARCHAR(50) NULL,
                       `candidate level` VARCHAR(50) NULL,
                       Experience TEXT NULL,
                       skill TEXT NULL,
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

