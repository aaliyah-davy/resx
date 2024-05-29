# IMAGE ADJUSTMENTS
import cv2
import numpy as np
import pandas as pd
from PIL import *
from itertools import groupby as gby
from IPython.display import display, HTML, Math

# SHOWME
from matplotlib import *
from matplotlib import mathtext
import matplotlib.pyplot as plt


def proxim (n, list_, minmax):
    """
    n: int
    list_: list of int
    minmax: str, either 'max' or 'min'

    returns values of list_ where n is the upper boundary ('max') or lower boundary ('min')
    """
    if minmax=='max':
        return max([l for l in list_ if l <= n] or [None])
    if minmax=='min':
        return min([l for l in list_ if l >= n] or [None])
    

def showme(imray):
    plt.imshow(imray,cmap='gray')
    plt.axis('off')
    plt.axes.format_coord = lambda x, y: ""


def TeX_print(predictions):    
    display(HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
                 "latest.js?config=default'></script>"))
    table = r"\begin{array} {l|l} %s \end{array}"
    return Math(table%'\\\\'.join(predictions))


def find_consecs(list_):
    """
    list_: list of int

    returns filtered values of list_ where each value is grouped if they are consecutive
    """
    gb = gby(enumerate(list_), key=lambda x: x[0] - x[1])
    all_groups = ([i[1] for i in g] for _, g in gb)
    ret = list(filter(lambda x: len(x) > 1, all_groups))
    return ret


def cpad(image,l,r,t,b):
    """
    l,r,t,b: left,right,top,bottom
    image: numpy array

    returns padded image based on specified l,r,t,b pad values
    """
    if len(image.shape)==2:
        padl = np.full((image.shape[0],l),255,dtype=np.uint8)
        image = np.hstack((padl,image))
        padr = np.full((image.shape[0],r),255,dtype=np.uint8)
        image = np.hstack((image,padr))
        padt = np.full((t,image.shape[1]),255,dtype=np.uint8)
        image = np.vstack((padt,image))
        padb = np.full((b,image.shape[1]),255,dtype=np.uint8)
        image = np.vstack((image,padb))
    if len(image.shape)==3:
        padl = np.full((image.shape[0],l,image.shape[2]),255,dtype=np.uint8)
        image = np.hstack((padl,image))
        padr = np.full((image.shape[0],r,image.shape[2]),255,dtype=np.uint8)
        image = np.hstack((image,padr))
        padt = np.full((t,image.shape[1],image.shape[2]),255,dtype=np.uint8)
        image = np.vstack((padt,image))
        padb = np.full((b,image.shape[1],image.shape[2]),255,dtype=np.uint8)
        image = np.vstack((image,padb))
    return image


def nxt(lst, element):
    """
    lst: list
    element: list of int

    returns the next element in lst, returns element if element is last
    """
    idx = lst.index(element)
    if idx!=len(lst)-1:
        return lst[idx+1]
    else:
        return lst[idx]
    

def get_vcoln(image):
    """
    image: numpy array

    returns pandas dataframe of indexes where columns of white pixels start
        * useful for multi-column text and identifying where text/figure/table ends
        * if only one column, only one index will be in the dataframe
    """
    vtrack = []
    for i in range(len(image[0,:])-1):
        if np.unique(image[:,i])[0]==255:
            vtrack.append(i)
    vtrack = pd.DataFrame(vtrack, columns = ['white_lines'])
    vtrack = vtrack[(vtrack['white_lines'] - vtrack.shift(1)['white_lines'] !=1)]
    vtrack = vtrack.reset_index(drop=True).astype('Int64')
    return vtrack


def get_hcoln(image):
    """
    image: numpy array

    returns pandas dataframe of indexes where rows of white pixels start
        * useful for identifying where text/figure/table ends
        * if only one row, only one index will be in the dataframe
    """
    htrack = []
    for i in range(len(image[:,0])-1):
        if np.unique(image[i,:])[0]==255:
            htrack.append(i)
    htrack = pd.DataFrame(htrack, columns = ['white_lines'])
    htrack = htrack[(htrack['white_lines'] - htrack.shift(1)['white_lines']!=1)]
    htrack = htrack.reset_index(drop=True).astype('Int64')
    return htrack


def inv_vcoln(image):
    """
    image: numpy array

    returns pandas dataframe of indexes where columns of non-white pixels start
        * useful for identifying where text/figure/table starts
        * if only one column, only one index will be in the dataframe
    """
    vtrack = []
    for i in range(len(image[0,:])-1):
        if np.unique(image[:,i])[0] != 255:
            vtrack.append(i)
    vtrack = pd.DataFrame(vtrack, columns = ['inv_lines'])
    vtrack = vtrack[(vtrack['inv_lines'] - vtrack.shift(1)['inv_lines']!=1)]
    vtrack = vtrack.reset_index(drop=True).astype('Int64')
    return vtrack


def inv_hcoln(image):
    """
    image: numpy array

    returns pandas dataframe of indexes where rows of non-white pixels start
        * useful for identifying where text/figure/table starts
        * if only one row, only one index will be in the dataframe
    """
    htrack = []
    for i in range(len(image[:,0])-1):
        if np.unique(image[i,:])[0] != 255:
            htrack.append(i)
    htrack = pd.DataFrame(htrack, columns = ['inv_lines'])
    htrack = htrack[(htrack['inv_lines'] - htrack.shift(1)['inv_lines']!=1)]
    htrack = htrack.reset_index(drop=True).astype('Int64')
    return htrack


def inv_list(imray):
    """
    imray: numpy array

    returns pandas dataframe of indexes where columns of non-white pixels start
        * unlike inv_vcoln, it thresholds the image first
        * useful for document images with sparse non-white pixels
    """
    edg = cv2.Canny(imray.copy(), 30, 200)
    kern = np.ones((1,8), np.uint8)  # horizontal kernel
    dil = cv2.bitwise_not(cv2.dilate(edg,kern, iterations=1))
    inv = inv_vcoln(dil)['inv_lines'].tolist() + [None]
    return inv


def get_bbox(imray):
    tl = inv_hcoln(imray)['inv_lines'][0],inv_vcoln(imray)['inv_lines'][0]
    br = get_hcoln(imray)['white_lines'].max(),get_vcoln(imray)['white_lines'].max()
    if br[0] == 0:
        br = imray.shape[0], br[1]
    if br[1] == 0:
        br = br[0], imray.shape[1]
    tlbr = tl,br
    return tlbr


def marg_pad(image):
    pad_size = 5
    if len(image.shape)==2:
        pady = np.full((pad_size,image.shape[1]),255,dtype=np.uint8)
        image = np.vstack((image,pady))
        pady = np.full((pad_size,image.shape[1]),255,dtype=np.uint8)
        image = np.vstack((pady,image))
        padx = np.full((image.shape[0],pad_size),255,dtype=np.uint8)
        image = np.hstack((image,padx))
        padx = np.full((image.shape[0],pad_size),255,dtype=np.uint8)
        image = np.hstack((padx,image))
    if len(image.shape)==3:
        pady = np.full((pad_size,image.shape[1],image.shape[2]),255,dtype=np.uint8)
        image = np.vstack((image,pady))
        pady = np.full((pad_size,image.shape[1],image.shape[2]),255,dtype=np.uint8)
        image = np.vstack((pady,image))
        padx = np.full((image.shape[0],pad_size,image.shape[2]),255,dtype=np.uint8)
        image = np.hstack((image,padx))
        padx = np.full((image.shape[0],pad_size,image.shape[2]),255,dtype=np.uint8)
        image = np.hstack((padx,image))
    return image


def tite(imray):
    imray = marg_pad(imray) # KEEP THIS!!
    (y0,x0),(y1,x1) = get_bbox(imray)
    imray = imray[y0:y1,x0:x1]
    return imray


def get_rects(imlist,dpi):
    ray = imlist
    rects = []
    for r in ray:
        temp = []
        img = np.array(r)
        kn = int(dpi/20)
        kernel = np.ones((kn,kn),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        edged = cv2.Canny(img.copy(), 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                       cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            temp.append(cv2.boundingRect(cnt))
        rects.append(temp)
        
    return rects

def get_margs(rects):
    mrg = 0
    for i,r in enumerate(rects):
        for ii,e in enumerate(rects):
            if i != ii:
                bewl = [(g,i) for g in r for o in e if g==o]
                if bewl != [] and mrg==0:
                    mrg = bewl
                if bewl != [] and mrg!=0:
                    mrg = mrg + bewl    
                    
    return mrg

def sort_margs(mrg,imlist):
    ray = [np.array(im) for im in imlist]
#     ray = imlist
    tups = [m for n in mrg for m in n if type(m)==tuple]
    tdf = pd.DataFrame(tups,columns=['x','y','w','h'])
    igr = [m for n in mrg for m in n if type(m)==int]
    mgx,mgy = [],[]
    for i in range(len(tups)):
        (x,y,w,h),ii = tuple(tdf.iloc[i]),igr[i]
        shp=ray[ii].shape
        if w*h>50 and np.unique(ray[ii][y:y+h+1,x:x+w+1])[0]!=255:
            if (x,y,w,h) not in mgx and h>w and x<shp[1]/10 or x>shp[1]-shp[1]/10:
                mgx.append((x,y,w,h))
            if (x,y,w,h) not in mgy and w>h and y<shp[0]/10 or y>shp[0]-shp[0]/10:
                mgy.append((x,y,w,h))
    return mgx,mgy,tups,igr

def marg_dfs(mgx,mgy,tups,igr):
    
    def get_pgs(marg_df):
        pg = 0
        if marg_df.empty==False:
            mask = [marg_df.iloc[i].values for i in range(len(marg_df))]
            idx = [t for t in range(len(tups)) for m in mask \
                 if tups[t][0]-5<=m[0]<=tups[t][0]+5 and \
                  tups[t][1]-5<=m[1]<=tups[t][1]+5 and \
                  tups[t][2]-5<=m[2]<=tups[t][2]+5 and \
                  tups[t][3]-5<=m[3]<=tups[t][3]+5 \
                 ]
            pg = list(set(np.array(igr)[idx]))

        return pg
    
    def redun(marg_df,xy):
        if xy=='y':
            nxy = 'h'
        if xy=='x':
            nxy = 'w'
        idx = [m for m in range(len(marg_df)) for g in range(len(marg_df)) \
                if marg_df.iloc[m][xy]>=marg_df.iloc[g][xy] and \
                marg_df.iloc[m][xy]+marg_df.iloc[m][nxy]<=marg_df.iloc[g][xy]+ \
                marg_df.iloc[g][nxy] if m != g]
        if idx != []:
            marg_df = marg_df[marg_df.index != idx[0]].reset_index(drop=True)
        return marg_df

    mgx = pd.DataFrame(mgx, columns=['x','y','w','h']).sort_values(['x'])
    mgx['gr'] = mgx['x'].diff().gt(20).cumsum()
    marg_x = mgx.groupby(['gr']).agg({'x':'min','y':'min','w':'max','h':'max'})
    marg_x = redun(marg_x,'x')
    pgx = get_pgs(marg_x)

    mgy = pd.DataFrame(mgy,columns=['x','y','w','h']).sort_values(['y'])
    mgy['gr'] = mgy['y'].diff().gt(20).cumsum()
    marg_y = mgy.groupby(['gr']).agg({'x':'min','y':'min','w':'max','h':'max'})
    marg_y = redun(marg_y,'y')
    pgy = get_pgs(marg_y)
        
    return marg_x,marg_y,pgx,pgy


def marg_coords(marg_x,marg_y):
    y1,y2 = [marg_y.iloc[0]['y'] + marg_y.iloc[0]['h'] if len(marg_y)>=1 else None][0], \
            [marg_y.iloc[1]['y'] if len(marg_y)>1 else None][0]
    x1,x2 = [marg_y.iloc[0]['x'] + marg_y.iloc[0]['w'] if len(marg_x)>=1 else None][0], \
            [marg_y.iloc[1]['x'] if len(marg_x)>1 else None][0]
    return x1,y1,x2,y2


def rm_margins(imlist,dpi):
    rects = get_rects(imlist,dpi)
    mrg = get_margs(rects)
    mgx,mgy,tups,igr = sort_margs(mrg,imlist)
    marg_x,marg_y,pgx,pgy = marg_dfs(mgx,mgy,tups,igr)
    x1,y1,x2,y2 = marg_coords(marg_x,marg_y)
    return x1,y1,x2,y2,pgx,pgy
