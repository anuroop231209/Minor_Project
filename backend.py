import os
import datetime
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


try:
    connection = pymysql.connect(**DB_CONFIG)
    print("Connected to MySQL successfully")

    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE();")
        result=cursor.fetchone()
        print(result)

        connection.close()

except Exception as e:
    print("Error:",e)