from docimtables import *

import cv2
import numpy as np
import pandas as pd
from statistics import *
from pytesseract import *
from pix2tex.cli import *
from PIL import *


# REQ'D! KEEP! DO NOT DELETE!!
def special_vcoln(image):
    img = image.copy()
    hcoln = get_hcoln(img)
    vcoln = get_vcoln(img)
    vemp, idx = [],[]

    for i in range(len(hcoln)-1):
        if i != len(hcoln)-1:
            temp = img[hcoln.loc[i][0]:].copy()
            vtemp = get_vcoln(temp)
            inn = list(vtemp.itertuples(index = False, name = None))
            if len(vtemp)<5 and inn not in vemp and not vcoln.equals(vtemp):
                vemp.append(inn)
                idx.append(i)

    vemp = pd.DataFrame([v for vem in vemp for ve in vem for v in ve],\
                        columns = ['white_lines'])
    return vemp, idx


# REQ'D! KEEP! DO NOT DELETE
def pad_imlist(imlist):
    list_=[]
    for r in imlist:
        diff=max([len(r[0,:]) for r in imlist])-len(r[0,:])
        if diff != 0:
            d=int(diff/2)
            if diff%2==0:
                padim=np.hstack((r,np.full((r.shape[0],d,3),(255,255,255))))
                padim=np.hstack((np.full((r.shape[0],d,3),(255,255,255)),padim))
                list_.append(padim)
            else:
                padim=np.hstack((r,np.full((r.shape[0],d,3),(255,255,255))))
                padim=np.hstack((np.full((r.shape[0],d+1,3),(255,255,255)),padim))
                list_.append(padim)
        else:
            padim=r
            list_.append(padim)
    
    imim = None
    mcolw=np.unique([im.shape[1] for im in list_])
    for m,o in enumerate(mcolw):
        cnt=0
        for i,im in enumerate([im for im in list_]):
            if im.shape[1]==mcolw[m]:
                if cnt==0:
                    imim=im
                    cnt+=1
                else:
                    imim=np.vstack((imim,im))
            
    return imim


# REQ'D! KEEP! DO NOT DELETE
def txtnimg(pgimg):
#     pgimg,tables = extract_tables(pgimg)
    special = special_vcoln(pgimg)
    hcoln = get_hcoln(pgimg)
    pgimg = [[[pgimg[:hcoln.loc[s][0]+5, :], pgimg[hcoln.loc[s][0]-5:, :]] \
            for s in special[1]][0] if special[0].empty==False else [pgimg]][0]
    
    lmno=0
    ims,txt,tables=[],[],[]
    for pim in pgimg:
        vcoln=get_vcoln(pim)
        imrays,text=[],[]
        for l in range(len(vcoln)-1):
            if l != len(vcoln)-1:
                col,tab=extract_tables(pim[:,vcoln.loc[l][0]:vcoln.loc[l+1][0]+10])
                tables+=tab
            else:
                col,tab=extract_tables(pim[:,vcoln.loc[l][0]+10:])
                tables+=tab
            
            hcoln = get_hcoln(col)
            for i in range(len(hcoln)-1):
                if i != len(hcoln)-1:
                    temp = col[hcoln.loc[i][0]:hcoln.loc[i+1][0]].copy()
                    if inv_hcoln(temp).empty==False and inv_vcoln(temp).empty==False:
                        temp = temp[inv_hcoln(temp)['inv_lines'][0]-1:,\
                                inv_vcoln(temp)['inv_lines'][0]-1:]
                    edged = cv2.Canny(temp, 30, 200)
                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                                           cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 0:
                        vtrack=[]
                        for v in range(len(temp[0,:])-1):
                            if np.unique(temp[:,v])[0]==255:
                                vtrack.append(v)
                                
                        if i<len(hcoln)-2 and len(hcoln)>2:
                            nemp = col[hcoln.loc[i+1][0]:hcoln.loc[i+2][0]].copy()
                            nedged = cv2.Canny(nemp, 30, 200)
                            nontours, _ = cv2.findContours(nedged, cv2.RETR_EXTERNAL, \
                                                           cv2.CHAIN_APPROX_NONE)
                            nont_thresh=[max([len(n) for n in nontours]) \
                                         if len(nontours) != 0 else 0][0]

                        cont_thresh=max([len(c) for c in contours])
                        cont_area=max([cv2.contourArea(cnt) for cnt in contours])

                        if cont_thresh>800 or (len(vtrack)>200 and nont_thresh>800)\
                        or cont_area>1000:
                            imrays.append(temp)
                        if cont_thresh<800 and cont_area<1000:
                            text.append(marg_pad(temp))
                else:
                    temp = col[hcoln.loc[i][0]:].copy()
                    edged = cv2.Canny(temp, 30, 200)
                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                                           cv2.CHAIN_APPROX_NONE)
                    if len(contours) != 0:
                        vtrack=[]
                        for v in range(len(temp[0,:])-1):
                            if np.unique(temp[:,v])[0]==255:
                                vtrack.append(v)

                        cont_thresh=max([len(c) for c in contours])
                        cont_area=max([cv2.contourArea(cnt) for cnt in contours])
                        
                        if cont_thresh>800 or cont_area>1000:
                            imrays.append(temp)
                        if cont_thresh<800 and cont_area<1000:
                            text.append(marg_pad(temp))
                            
                            
        txim = pad_imlist(text)  
        if len(txim) != 0:
            txt.append(txim)

        imim = pad_imlist(imrays)
        if len(imrays) != 0:
            ims.append(imim)

    txt = pad_imlist(txt)
    ims = pad_imlist(ims)
            
    return txt,ims,tables