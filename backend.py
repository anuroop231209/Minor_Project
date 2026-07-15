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