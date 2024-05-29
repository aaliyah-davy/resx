from resx_init import *


import cv2
import math
import imutils
import numpy as np
import pandas as pd
from statistics import *
from pytesseract import *
from pix2tex.cli import *
from PIL import *


def dotted(img):
    ret,img = cv2.threshold(img.copy(),127,255,cv2.THRESH_TOZERO)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    lines=[li[0] for li in lines]

    df,dfx=[],[]
    if len(lines) > 0:
        for l,line in enumerate(lines):
            x0, y0, x1, y1 = np.round(line[0]),np.round(line[1]),\
                            np.round(line[2]),np.round(line[3])
            if y0==y1:
                df.append((x0, y0, x1, y1))

            if x0==x1:
                dfx.append((x0, y0, x1, y1))

    df = pd.DataFrame(df, columns = ['x1','y1','x2','y2']).sort_values(['y1','x1'])
    df['gr'] = df.diff().gt(5).cumsum().groupby(['y1']).ngroup()
    df = df.astype('Int64')
    sz = df.groupby(['gr']).size()
    sz = sz[sz>50].reset_index(name='sz')
    mask = df[df['gr'].isin(sz.gr)]
    mask = mask.groupby(['gr'])['y1'].min().tolist()
    
    return mask

def fill_dotted(img):
    img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
    shp = [img.shape[2] if len(img.shape)>2 else None][0]
    bw = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    dots = dotted(bw)

    if len(dots) > 0:
        for i in range(len(dots)):
            y = dots[i]
            x1,x2 = inv_vcoln(img[y:y+2])['inv_lines'].min(),\
                    inv_vcoln(img[y:y+2])['inv_lines'].max()
            if shp is None:
                img[y+1:y+3,x1:x2] = np.ones((y-y+2,x2-x1),np.uint8)
            else:
                img[y+1:y+3,x1:x2] = np.ones((y-y+2,x2-x1,shp),np.uint8)
    return img


# Col-Wise
def tab_x(imray):
    img = imray.copy()
    img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
    kern = np.ones((3,1), np.uint8)  # vertical kernel
    img = cv2.dilate(img,kern, iterations=1)

    sh = img.shape[1]/1000
    mx = int(math.modf(sh)[1])+2

    ch,x_lines = [],[]
    for m in range(1,mx):
        st = [1000*m - ch[-1] if len(ch)>0 and m>1 else 0][0]*(m-1)
        sp = [1000*m if len(ch)>0 and mx-1>m>1 else 1000 if len(ch)==0 else None][0]
        ch.append(sp)
        im_ = img[:,st:sp]

        if st < img.shape[1]:
            for i in range(len(im_[0,:])):
                inpl = im_[:,i:i+5]
                edged = cv2.Canny(inpl.copy(), 30, 200)
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                    cont_thresh=max([len(c) for c in contours])
                    if cont_thresh>200:
                        x_lines.append(st+i)
                        
    x_lines = pd.DataFrame(x_lines, columns = ['x1']).sort_values(['x1'])
    x_lines['gr'] = x_lines.diff().gt(5).cumsum().groupby(['x1']).ngroup()
    x_lines = x_lines.sort_values(['x1'])
    x_lines = x_lines.groupby('gr').agg({'x1':'min'})
    
    return x_lines


# ROW-WISE LINES FOR TABLE-FINDER
def tab_y(imray):
    img = imray.copy()
    img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
    kern = np.ones((1,3), np.uint8)  # horizontal kernel
    img = cv2.dilate(img,kern, iterations=1)

    ch,y_lines = [],[]
    sh = img.shape[0]/1000
    mx = int(math.modf(sh)[1])+2
    for m in range(1,mx):
        st = [1000*m - ch[-1] if len(ch)>0 and m>1 else 0][0]*(m-1)
        sp = [1000*m if len(ch)>0 and mx-1>m>1 else 1000 if len(ch)==0 else None][0]
        ch.append(sp)
        im_ = img[st:sp,:]
        if st < img.shape[0]:
            for i in range(len(im_[0,:])):
                inpl = im_[i:i+3,:]
                edged = cv2.Canny(inpl.copy(), 30, 200)
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_NONE)
                if len(contours) != 0:
                    cont_thresh=max([len(cnt) for cnt in contours])
#                     cont_area=max([cv2.contourArea(cnt) for cnt in contours])
                    if cont_thresh>200:
                        y_lines.append(st+i)
                        
    y_lines = pd.DataFrame(y_lines, columns = ['y1']).sort_values(['y1'])
    y_lines['gr'] = y_lines.diff().gt(5).cumsum().groupby(['y1']).ngroup()
    y_lines = y_lines.sort_values(['y1'])
    y_lines = y_lines.groupby('gr').agg({'y1':'min'})
    
    return y_lines


def lgr_cond(imray,dfy):
    lgr, x1s = [], []
    
    for d in dfy['y1'].tolist():
        bw = cv2.cvtColor(imray.copy(), cv2.COLOR_BGR2GRAY)
        img = bw[d:d+3]
        kern = np.ones((1,3), np.uint8)  # horizontal kernel
        img = cv2.dilate(img, kern, iterations=1)
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img)[0]

        if lines is not None:
            x1,y1,x2,y2 = [li[0] for li in lines if li[0][3]==li[0][1]][0].astype(int)
            lgr.append([True if abs(x2-x1)>img.shape[1]/10 else False][0])
            x1s.append(y1)
            
    dfy['lgr'], dfy['x1'] = lgr, x1s
    dtl = dfy[dfy['lgr'] != False].drop(columns = ['lgr']).reset_index(drop=True)
    
    return dtl


def typical_table(imray,dtl):
    img = imray.copy()
    coords = []
    y1s = dtl['y1'].tolist()
    mask = inv_hcoln(img)['inv_lines'], get_hcoln(img)['white_lines']
    tb = [(m,[n for n in mask[1] if n>m][0]) for m in mask[0] for y in y1s if y-5<m<y+5]
    idx = [m[0] for m in enumerate(mask[0]) for y in y1s if y-5<m[1]<y+5][0]
    x1 = proxim(mask[0][idx], y1s,'min')
    for t in tb:
        y1,y2 = t[0],t[1]
        msk = get_vcoln(img[y1:y2+5,x1:])['white_lines']
        x2 = [x+x1 for x in msk if x+x1>x1][0]
        coords.append((x1,y1,x2,y2))
        
    return coords


def trendy_table(imray,dtl):
    img = imray.copy()
    
    x2s = []
    for d in dtl['y1'].tolist():
        x1 = inv_vcoln(img[d:d+5])['inv_lines'][0]
        mask = get_vcoln(img[d:d+5,x1:])['white_lines']
        x2 = [x+x1 for x in mask if x+x1>x1][0]
        x2s.append(x2)

    x2 = pd.DataFrame(x2s, columns = ['x2']) # ['x1','y1','x2','y2']
    cat = pd.concat([dtl, x2, dtl.rename(columns = {'y1':'y2'})['y2']],\
                                                axis=1, join="inner")
    cat = cat[['x1','y1','x2','y2']]
    return cat


def tab_cols(df,ogim):
    x1s = [int(x1) for x1 in df['x1'].tolist()]
    y1s = [int(y1) for y1 in df['y1'].tolist()]
    x2s = [int(x2) for x2 in df['x2'].tolist()]
    y2s = [int(y2) for y2 in df['y2'].tolist()]

    inv = []
    vch = []
    for i in range(len(y2s)-1):
        lp = 5
        inpl = ogim[y2s[i]+lp:y2s[i+1],x1s[i]:x2s[i]]
        ret,thresh = cv2.threshold(inpl,127,255,cv2.THRESH_TOZERO)
        edged = cv2.Canny(thresh.copy(), 30, 200)
        kernel = np.ones((25, 25), np.uint8)
        dil = cv2.dilate(edged, kernel, iterations=1)
        contours = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
        cons = marg_pad(cv2.bitwise_not(dil))
        vcoln = inv_vcoln(cons)
        if i>0 and x2s[i]-10 <= x2s[i-1] <= x2s[i]+10 and\
                    x1s[i]-10 <= x1s[i-1] <= x1s[i]+10 and len(vcoln) < len(vch[-1]):
            vcoln.loc[-1] = 0
            vcoln.index = vcoln.index + 1
            vcoln = vcoln.sort_index()
        inv.append(vcoln)
        vch.append(vcoln)

    mask = [i for i in range(max([len(v) for v in inv]))]
    inv = pd.concat([v.T for v in inv])\
            .rename(columns = {int('{}'.format(m)):'col_{}'.format(m) for m in mask})
    inv['gr'] = inv.count(axis = 1).diff().fillna(0).ne(0).cumsum().tolist()
    inv = inv.astype('Int64')

    igr = inv['gr'].tolist()
    df['gr'] = [0]+[g if igr.count(g)>1 else nxt(igr,g) for g in igr]
    return df

def df_bbox(df):
    bbox=[]
    for i in df['gr'].unique():
        box=df[df.gr==i]['x1'].min(),df[df.gr==i]['y1'].min()-5,\
            df[df.gr==i]['x2'].max(),df[df.gr==i]['y2'].max()+5
        bbox.append(box)
    return bbox

def extract_tables(image):
    ogn = image.copy()
    shp = [ogn.shape[2] if len(ogn.shape)>2 else None][0]
    img = fill_dotted(image)
    dx1, dy1 = tab_x(img), tab_y(img)
    dy1 = lgr_cond(img,dy1)
    
    try:
        if len(dx1)>1 and len(dy1)>1:
            tab_box = typical_table(img, dy1)
        if len(dx1)<2 and len(dy1)>1:
            df = trendy_table(img,dy1)
            df = tab_cols(df, image)
            tab_box = df_bbox(df)
        else:
            tab_box = []
    except:
        tab_box = []
        
    try:
        repl, itemp = [(tab_box[i][3],i) for i in range(len(tab_box)-1) if \
               tab_box[i+1][3]-20 < tab_box[i][3] < tab_box[i+1][3]+20][0]
        tab_box = tab_box[:itemp] + [(tab_box[itemp][0],tab_box[itemp][1], \
                    tab_box[itemp][2], tab_box[itemp+1][3])] + \
                    tab_box[itemp+2:]
    except:
        pass

    try:
        repl, itemp = [(tab_box[i][1],i) for i in range(len(tab_box)-1) if \
               tab_box[i+1][1]-20 < tab_box[i][1] < tab_box[i+1][1]+20][0]
        tab_box = tab_box[:itemp] + [(tab_box[itemp][0],tab_box[itemp][1], \
                tab_box[itemp][2], tab_box[itemp+1][3])] + \
                tab_box[itemp+2:]
    except:
        pass
    
    tables = []
    if len(tab_box) > 0:
        for tab in tab_box:
            x1,y1,x2,y2 = tab
            tables.append(ogn[y1-5:y2, x1:x2])
            if shp is None:
                image[y1-5:y2, x1:x2] = np.full((y2-y1+5, x2-x1), 255)
            else:
                image[y1-5:y2, x1:x2] = np.full((y2-y1+5, x2-x1, shp), 255)
            
    return image, tables 
