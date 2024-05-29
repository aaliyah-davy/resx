from docimfigs import *

# remove_images + download_pdf_file
import os
import requests

# GET PDFIMG
import imgkit
from io import BytesIO
from sys import platform
from pdf2image import convert_from_bytes, convert_from_path
from fake_useragent import UserAgent

import cv2
import numpy as np
from statistics import *
from pytesseract import *
from pix2tex.cli import *
from PIL import *


def get_rilines(inn,dpi):
    if isinstance(inn,str):
        ray = convert_from_path(inn, fmt='PNG', dpi=dpi) #mdl data dpi=250
    if isinstance(inn,bytes):
        ray = convert_from_bytes(inn, fmt='PNG', dpi=dpi)
    x1,y1,x2,y2,pgx,pgy = rm_margins(ray,dpi)
    PAGE = 3
    img = np.array(ray[pgy[PAGE]])[y1:y2,x1:x2]
    pgimg = marg_pad(img.copy())
    rett, reti, tables = txtnimg(pgimg)
    rett = rett.copy().astype(np.uint8)
    hcoln = get_hcoln(rett.copy())['white_lines'].tolist()
    rilines = [rett[hcoln[i]:hcoln[i+1]] for i in range(len(hcoln)-1)]
    return rilines


def download_pdf_file(url):
    response = requests.get(url, stream=True)

    pdf_file_name = os.path.basename(url)
    if response.status_code == 200:
        # filepath = os.path.join(os.getcwd(), pdf_file_name)
        buffer=BytesIO()
        buffer.seek(0)
        with buffer as b:
            b.write(response.content)
            out_buffer=b.getvalue()
            return out_buffer
    else:
        print(f'Uh oh! Could not download {pdf_file_name},')
        print(f'HTTP response status code: {response.status_code}')



def get_pdfimg(url):
    ua = UserAgent()

    if platform == "linux" or platform == "linux2" or platform == "darwin":
        path_wkhtmltopdf = r'/usr/local/bin/wkhtmltopdf'
    elif platform == "win32":
        path_wkhtmltopdf = r"C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"
    config = imgkit.config(wkhtmltoimage=path_wkhtmltopdf)
    options = {
        'custom-header': [('User-Agent', ua.random)],
    }
    try:
        buf = BytesIO()
        buf.seek(0)
        with buf as f:
            f.write(imgkit.from_url(url,False,config=config,options=options))
            buffy=f.getvalue()
            # out_buffer=remove_images(buffy)
            pdf_=convert_from_bytes(buffy,fmt='jpeg')
    except:
        outs=download_pdf_file(url)
        # outs=remove_images(retry)
        pdf_=convert_from_bytes(outs,fmt='jpeg')


    for i in range(len(pdf_)):
        pdf_[i]=np.array(pdf_[i],dtype='uint8')
        pdf_[i] = cv2.cvtColor(pdf_[i], cv2.COLOR_BGR2GRAY)
    
    return pdf_

