# import 
from docim2txt import *

# get_audio
import pandas as pd
import cv2
import numpy as np
import torch
from io import BytesIO
from TTS.api import *
import matplotlib.pyplot as plt
from pydub import AudioSegment

# text processing
import regex as re

# get_frame
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# get_clip
from moviepy.editor import *
from moviepy.audio.AudioClip import *
from scipy.io.wavfile import read as wav_read


def get_sntarr(section):
    section = re.sub(r'\s\[(.*?)\].', '. ', section)
    pattern = '(?<=[a-z]+\.\s)([A-Z])|(?<=[a-z]+\.)([A-Z][a-z]+)'
    itr = re.finditer(pattern, section)
    idx = [0] + [m.start(0) for m in itr] + [len(section)]
    sntarr = [section[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    return sntarr

def get_audio(sntarr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name="tts_models/en/ljspeech/tacotron2-DDC"

    itr=0
    for snt in sntarr:
        buffer=BytesIO()
        buffer.seek(0)
        with buffer as b:
            TTS(model_name).to(device).tts_to_file(text=snt, file_path=b)
            if itr==0:
                audio = AudioSegment.from_file(BytesIO(b.read()), format="wav")
                itr+=1
            else:
                a2 = AudioSegment.from_file(BytesIO(b.read()), format="wav")
                audio += a2
    return audio

def get_txt_wdw(wdw_max, section, raw_snt):
    snt_len = [len(snt) for snt in raw_snt]
    
    result = 0
    txt_wdw = []
    temp = ''
    
    for v in range(len(snt_len)):
        if result > wdw_max:
            temp = temp[:temp.find(raw_snt[v-1])]
            txt_wdw.append(temp)
            temp = raw_snt[v-1]
            result = len(raw_snt[v-1])
            
        if v==len(snt_len)-1:
            temp += raw_snt[v]
            txt_wdw.append(temp)
            
        temp += raw_snt[v]
        result += snt_len[v]
        
    return txt_wdw

def csp(wndw, step, start):
    prev_spce = step - wndw[start:start+step][::-1].find(' ') - 1
    end = start+prev_spce
    wndw = wndw[start:end]
    return end,wndw

def get_txt_line(raw_txt, step):
    end = -1
    txt_line = []
    for start in range(0, len(raw_txt), step):
        start = end+1
        end,wndw = csp(raw_txt, step, start)
        txt_line.append(wndw)
    _,last = csp(raw_txt, step, end+1)
    txt_line.append(last)
    return txt_line

def get_frame(img, txt_wdw, line_max, sect_name, cite):
    txt_line = get_txt_line(txt_wdw, line_max)
    shp = np.shape(img.transpose())
    img = img.transpose(2,0,1)

    pad_=int((650-(shp[1]))/2) # 16:9 ratio for videos
    pads = ((pad_,pad_),(int(2*pad_/3),int(pad_*8))) # ((t,b),(l,r))
    lpad,rpad=np.shape(img)[1]+sum(pads[0]),\
            np.shape(img)[2]+sum(pads[1])

    img_arr = np.ndarray((3,lpad,rpad),int)
    for i,x in enumerate(img):
        img_arr[i,:,:] = np.pad(x,pads,'constant',constant_values=0) # 255 is black

    img_arr = np.uint8(img_arr).transpose(1,2,0)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img) 
    x,y = img.size[1],img.size[0]/7
    draw.line((x-x/25,y,x*1.65,y), fill="white")
    font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 19)
    abf = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 40)
    cite_font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 11)
    
    cx,cy = pads[1][0], tx-pads[1][0]+5
    draw.text((cx,cy,cx,cy),cite,(255,255,255),font=cite_font)
    tx,ty,tx2,ty2 = (x+9-x/25,y+10,x*1.65,y+50)
    
    for i in range(2):
        draw.text((tx,ty-75,tx2,ty2),sect_name,(255,255,255), font=abf)
    for line in txt_line:
        draw.text((tx,ty,tx2,ty2),line,(255,255,255), font=font)
        ty+=30
        
    frame = np.array(img)
    return frame

def get_clip(frame, wdw_audio):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    wdw_audio.export(tmp.name, format="wav")
    audio = AudioFileClip(tmp.name)
    tmp.close()
    
    clip = ImageClip(frame).set_duration(audio.duration)
    clip = clip.set_audio(audio)
    clip.fps=24
    return clip

def get_video(section, sect_name, imlist, cite):
    wdw_max, line_max = 600, 50

    raw_snt = get_sntarr(section)
    txt_wdw = get_txt_wdw(wdw_max, section, raw_snt)

    vid_arr = []
    for i in range(len(txt_wdw)):
       # im = get_im(txt_wdw[i]) # when i figure out google colab, change txt_wdw to just wdw
        frame = get_frame(imlist[i],txt_wdw[i],line_max,sect_name,cite) # uses get_text_line
        wdw_snt = get_sntarr(txt_wdw[i])
        wdw_audio = get_audio(wdw_snt)
        clip = get_clip(frame, wdw_audio)
        vid_arr.append(clip)

    video = concatenate_videoclips(vid_arr)
    return video