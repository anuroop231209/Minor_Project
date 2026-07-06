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
        reader=PdfReader(file_path)

        text=""

        for page in reader.pages:
            page_text=page.extract_text()

            if page_text:
                text+=page_text + "\n"

        return text
    
    except Exception as e:
        print("Error",e)
        return""

        
if __name__ == "__main__":
    pdf_path= "Uploaded_Resumes/2.pdf"

    text=extract_text_from_pdf(pdf_path)

    print("==Resume Text")
    print(text)