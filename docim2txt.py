from pdf2img import *

# remove_images + download_pdf_file
import io

import re
import cv2
import string
import numpy as np
import regex as re
from PIL import Image, ImageOps
import pandas as pd
from statistics import *
from pytesseract import *
from pix2tex.cli import *
from PIL import *
from statistics import mode
from pix2tex.cli import LatexOCR as model


def get_imTeX(imlist):
    txt = raw_tess(imlist)
    d = cat_tess(txt)
    d = d[(d.text != '')].sort_values(by=['im', 'left']).reset_index(drop=True)
    d = rm_incon(d,imlist)
    d = update_tess(d,imlist)
    indv_words, txt_list, full_text, sntc = seg_tess(d)
    return indv_words, txt_list, full_text, sntc


def raw_tess(imlist):
    txt = []
    i = 0
    for im in imlist:
        img = marg_pad(im.copy())
        pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
        config=r'--oem 3 -l eng --psm 6 -c tessedit_create_hocr=1'
        d = pytesseract.image_to_data(img, output_type=Output.DICT,config=config)
        txt.append((d['conf'],d['text'],d['left'],d['width']))
    return txt


def cat_tess(txt):
    cntr = 0
    for t in txt:
        df = pd.DataFrame(t).T.rename(columns = {0:'conf', 1:'text', \
                                                2: 'left', 3: 'width'})
        if cntr>0:
            d = pd.concat([d,df])
        if cntr==0:
            d = df
            cntr+=1

    d['im'] = d.groupby(d.index==0).ngroup().ne(0).cumsum()-1
    d = d.reset_index(drop=True)
    return d


def rm_incon(d,imlist): 
    ims = d.im.unique()
    for im in ims:
        img = imlist[im]
        edg = cv2.Canny(img.copy(), 30, 200)
        kern = np.ones((1,8), np.uint8)  # horizontal kernel
        dil = cv2.bitwise_not(cv2.dilate(edg,kern, iterations=1))
        inv = inv_vcoln(dil)['inv_lines'].tolist()
        comp = d[d.im==im]
        cf = np.any(d[d.im==im]['conf']<90)
        if len(inv) < len(comp) and cf == True:
            vcoln = get_vcoln(dil)['white_lines'].tolist()
            lfs,ix = d[d.im==im]['left'].tolist(),d[d.im==30].index.tolist()
            temp,rm = [],[]
            for i,v in enumerate(vcoln):
                for ii,l in enumerate(lfs):
                    if i!=len(vcoln)-1 and v <= l <= vcoln[i+1]:
                        if v in temp:
                            rm.append(ix[ii])
                            d = d.drop(index = ix[ii])
                        else:
                            temp.append(v)

    d = d.reset_index(drop = True)
    return d


def update_tess(d,imlist):
    repl = corr_tess(d,imlist)
    inpl = romans(d)
    print('repl:',len(repl),'inpl:',len(inpl))
    d.loc[inpl.index,'text'] = repl
    return d


def seg_tess(d):
    indv_words, txt_list, full_text = [], [], []
    for i in d.im.unique():
        words = d[(d.im==i) & ~(d.text=='')]['text'].values
        full = '\\mathrm{' + '~'.join(words) + '}'
        indv_words.append(words), txt_list.append(full)
    full_text = '~'.join(txt_list)
    
    d['sent'] = d.groupby(d.text.str.endswith('.')).ngroup().shift().\
                fillna(0).ne(0).cumsum()
    
    sntc = []
    for s in d.sent.unique():
        sentence =  '~'.join(d[(d.sent==s) & ~(d.text=='')]['text'].values)
        sentence =  '\\mathrm{' + sentence + '}'
        sntc.append(sentence)
        
    return indv_words, txt_list, full_text, sntc


def romans(d):
    rmn = r'(?=\b[MDCLXVI]+\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
    inpl = d[(d.conf<91) & (d.conf>=0) & \
           ~((d.text.str.contains('|'.join(filter(str.isalnum, rmn)))) & \
             (d.text.str.contains('|'.join('-')))) & \
            ~((d.text.str.contains('|'.join(filter(str.isalnum, rmn)))) & \
              (d.text.str.contains('|'.join('"')))) & \
             ~(d.text.str.contains(rmn,regex=True)) & \
             ~((d.text.str.match('^[a-zA-Z]+[”]$')) | \
               (d.text.str.match('^[“][a-zA-Z]+$')) | \
               (d.text.str.match('^[“][a-zA-Z]+[”]$')))]
    return inpl


def im_wndw(dx,d,rilines):
    img = rilines[d.im[dx]]
    edg = cv2.Canny(img.copy(), 30, 200)
    kern = np.ones((1,8), np.uint8)  # horizontal kernel
    dil = cv2.bitwise_not(cv2.dilate(edg,kern, iterations=1))
    inv = inv_vcoln(dil)['inv_lines'].tolist()
    vcoln = get_vcoln(dil)['white_lines'].tolist()
    start = proxim(d.left[dx],inv,'max')
    stop = proxim(d.left[dx],vcoln,'min')
    stop = [stop if stop>start else None][0]
    img = rilines[d.im[dx]][:,start:stop]
    img = tite(img)
    return img


def mod_er(text_seg):
    idx = [[t.start(),t.end()] for t in re.finditer('\{([^}]+)', text_seg)]
    s = text_seg

    for i in idx:
        f = text_seg[i[0]:i[1]]
        find = [m.start() for m in re.finditer(r'\\', f)]
        for fn in find:
            fn = fn+i[0]
            if s[fn:fn+7] != '\\mathrm' and  s[fn+1] != '~':
                s = s[:fn] + '~' + s[fn + len('\\'):]
    return s


def repl_slash(text):
    poss = string.punctuation
    non = [i for i,t in enumerate(text) if t.isalnum()==False]

    for i,txt in enumerate(text):
        if txt=='\\':
            st = nxt(non,i)
            if text[i:st] not in poss:
                text = text[:i] + '~' + text[i+1:]
    return text


def rm_dup(tcr_,p):
    for i,tc in enumerate(tcr_):
        til = [ii for ii,t in enumerate(tc) if t==p]
        cns = find_consecs(til)
        if cns != []:
            for c in cns:
                tcr_[i] = tcr_[i][:min(c)] + tcr_[i][max(c):]
    return tcr_


def try_blocks(inpl,d,rilines):
    txs = []
    idx = inpl.index.tolist()
    tmp = []
    checks = len(inpl)

    for dx in idx:
        rd,sz = 1.6,1
        img = im_wndw(dx,d,rilines)
        out = get_out(img,rd,sz,'yes') # get_out(dx,d,rilines,1.6)
        try:
            myb = TeX_2ray(r''+out)
            tmp.append(myb)
            txs.append(out)
        except:
            try:
                rd,sz = 1.8,1
                img = im_wndw(dx,d,rilines)
                out = get_out(img,rd,sz,'yes') # get_out(dx,d,rilines,1.8)
                myb = TeX_2ray(r''+out)
                tmp.append(myb)
                txs.append(out)
            except:
                try:
                    out = snt(out)
                    myb = TeX_2ray(r''+out)
                    tmp.append(myb)
                    txs.append(out)
                except:
                    tmp.append(out)
                    checks-=1
                    print(dx,out)
                    txs.append(out)

    if checks==len(inpl):
        print('All passed!')
    else:
        print('{} failed'.format(len(inpl)-checks),checks)
    return txs,tmp


def snt(text_seg):
    s = text_seg
    
    idx = [[t.start(),t.end()] for t in re.finditer(r'\\\w+', s)]
    for i in range(len(idx)):
        n,x = idx[i][0],idx[i][1]
        try:
            mathtext.math_to_image('$'+s[n:x]+'{temp}$','m')
            idx[i] = None
        except:
            pass
        
    idx = [i for i in idx if i is not None]
    pl = 0
    for i in idx:
        f,n = i[0]+pl,i[1]+pl
        if s[f:n] != '\\mathrm':
            s = s[:f] + '\\mathrm' + s[n:]
            pl+= len('\\mathrm') - n + f
            
    idx = [[t.start(),t.end()] for t in re.finditer('\w+', s)]
    pl = 0
    for i in idx:
        f,n = i[0]+pl,i[1]+pl
        if s[f:n] == 'bigg' or s[f:n] == 'Bigg' or \
            s[f:n] == 'big' or s[f:n] == 'Big':
            s = s[:f] + s[n:]
            pl+= n - f
    
    idx = [l for p in [[t.start(),t.end()] for t in re.finditer(r'\\', s)] for l in p]
    idx = find_consecs(list(np.unique(idx)))
    idx = [l for l in idx if len(l)>2]
    pl = 0
    if idx != []:
        for i in idx:
            n,x = min(i)+pl,[max(i) if max(i)>len(idx)-2 else max(i)+1][0]+pl
            s = s[:n] + '\\' + s[x:]
            pl+= len('\\') - x + n
                
    idx = [t.end() for t in re.finditer('([}]+)', s) if t.end()!=len(s)]
    for i in idx:
        if s[i].isalpha() == True and s[i-2] != '~':
            s = s[:i] + '~' + s[i:]
        if s[i].isalpha() == True and s[i-2] == '~':
            s = s[:i-2] + s[i-1:]
            s = s[:i-1] + '~' + s[i-1:]
              
    return s


def rm_punc(inn):
    import string
    ret = " ".join("".join([" " if ch in string.punctuation \
                            else ch for ch in inn]).split())
    return ret

def best(outs,dx,d):
    l = 100
    for o in outs:
        idx = [[t.start()+1,t.end()] for t in re.finditer(r'{\w+', o)]
        if idx != []:
            i,x = max(idx)
            b4 = rm_punc(d.iloc[dx-1]['text'])
            nx = rm_punc(d.iloc[dx+1]['text'])
            if nx in o and b4 in o and o[-1] != '|':
                if len(o) < l:
                    l = len(o)
                    ret = o
    return ret


def tite(imray):
    (y0,x0),(y1,x1) = get_bbox(imray)
    imray = imray[y0:y1,x0:x1]
    return imray


def get_out(img,rd,sz,blur):
    if blur=='yes':
        img = cv2.GaussianBlur(img.copy(), (sz, sz), 0) # Gaussian
    rx,ry = img.shape[1]/rd,img.shape[0]/rd
    im = Image.fromarray(img)
    orig_size = im.size
    im.thumbnail([rx, ry])
    im = im.transform(orig_size, Image.EXTENT, (0,0, rx, ry))
    out = run_5x(im)
    return out


def rep_this(text_seg,this,that):
    idx = [t.start() for t in re.finditer(this, text_seg)]
    s = text_seg

    while len(idx)>0:
        fn = idx[0]
        s = s[:fn] + that + s[fn + len(this):]
        idx = [t.start() for t in re.finditer(this, s)]
        
    return s


def TeX_2ray(formula):
    formula = r'$' + formula + '$'
    dpi = 400

    buf = io.BytesIO()
    mathtext.math_to_image(formula, buf, format="png", dpi=dpi)
    buf.seek(0)
    imray = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    imray = cv2.imdecode(imray, -1)
    
    imray = marg_pad(imray) # KEEP THIS!!
    (y0,x0),(y1,x1) = get_bbox(imray)
    imray = imray[y0:y1,x0:x1]
    return imray


def run_5x(img):
    model = LatexOCR()
    md,x = [],5
    while x != 0:
        md.append(model(img))
        x-=1
    md = '' + mode(md) + ''
    return md


def corr_tess(d,imlist):
    tcr = tess_err(d,imlist)
    list_ = seg_TeX(tcr)
    br = ['[ ]', '( )', '{ }']
    for b in br:
        list_ = bal_brck(list_,b)
    return list_


def run_4x(img):
    model = LatexOCR()
    md,x = [],4
    while x != 0:
        md.append(model(img))
        x-=1
    md = '' + mode(md) + ''
    return md


def tess_err(d,imlist):
    inpl = romans(d)
    tcr = []

    for i,pl in inpl.iterrows():
        stop = [None if i+2>max(d[d.im==pl.im].index) else \
                d.left[i+2]][0]
        img = imlist[pl.im][:,d.left[i-2]:stop].copy()
        img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
        sz = 2
        img = cv2.blur(img.copy(), (sz, sz))
        img = Image.fromarray(img)
        tcr.append(run_4x(img))
    return tcr


def punc_try(txt,punc):
    try:
        ret = txt.split(punc)[2]
    except:
        ret = None
    return ret


def seg_TeX(tcr):
    punc = ['~', ' ', ',', ';']
    seg = [punc_try(t,p) for t in tcr for p in punc if punc_try(t,p) is not None]
    return seg


def bal_brck(list_,br):
    lb, rb = br.split(' ')[0], br.split(' ')[1]
    cntr = 0
    for i,o in enumerate(list_):
        for ii,t in enumerate(o):
            if t==lb:
                if ii==len(list_[i])-1:
                    list_[i] = list_[i] + rb
                else:
                    cntr+=1
            if t==rb:
                if ii==0:
                    list_[i] = lb + list_[i]
                else:
                    cntr-=1
        while cntr > 0:
            list_[i] = list_[i] + rb
            cntr-=1
        while cntr < 0:
            list_[i] = lb + list_[i]
            cntr+=1
    return list_


def rep_this(text_seg,this,that):
    idx = [t.start() for t in re.finditer(this, text_seg)]
    s = text_seg

    while len(idx)>0:
        fn = idx[0]
        s = s[:fn] + that + s[fn + len(this):]
        idx = [t.start() for t in re.finditer(this, s)]
        
    return s


def the_repl(text_seg,imray):
    inpl = [space.start() for space in re.finditer('~', text_seg)]
    th = text_seg.find('[he')
    idx = inpl.index(proxim(th, inpl, 'min'))

    img = imray.copy()
    img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
    edg = cv2.Canny(img.copy(), 30, 200)
    kern = np.ones((1, 7), np.uint8)  # horizontal kernel
    dil = cv2.bitwise_not(cv2.dilate(edg, kern, iterations=1))
    vcoln = [0] + get_vcoln(dil)['white_lines'].tolist() + [None]
    sdx = [idx+2 if idx<=len(vcoln)-2 else \
             idx+1 if idx==len(vcoln)-1 else None][0]
    print(sdx,vcoln,len(vcoln))
    st,sp = vcoln[idx], [None if sdx is None else vcoln[sdx]][0]

    img = imray.copy()
    img = cv2.blur(img, (2, 1)) 
    img = img[:,st:sp]
    img = ImageOps.grayscale(Image.fromarray(img))
    new_text = model(img) + '~'
    text_seg = text_seg[:th] + new_text + text_seg[len(new_text):]
    return text_seg


def space_repl(text_seg,imray):

    stf = min([err.start() for err in re.finditer(' ', text_seg)])
    space = [err.start() for err in re.finditer('~', text_seg)]
    try:
        idx = space.index(proxim(stf,space,'max'))
    except:
        idx = 1

    img = imray.copy()
    img = (((img - img.min()) / (img.max()-img.min())) * 255).astype(np.uint8)
    edg = cv2.Canny(img.copy(), 30, 200)
    kern = np.ones((1, 7), np.uint8)  # horizontal kernel
    dil = cv2.bitwise_not(cv2.dilate(edg, kern, iterations=1))
    vcoln = [0] + get_vcoln(dil)['white_lines'].tolist() + [None]
    st = vcoln[idx-1]
    
    imray = imray.copy()
    imray = cv2.blur(imray, (1, 1)) 
    kernel = np.ones((1,3),np.uint8)
    imray = cv2.morphologyEx(imray, cv2.MORPH_OPEN, kernel)

    img = imray[:,st:]
    img = ImageOps.grayscale(Image.fromarray(img))

    if st is not None and st>0:
        pimg = imray[:,:st]
        pimg = ImageOps.grayscale(Image.fromarray(pimg))
        text_seg = model(pimg) + '~' + model(img)
    else:
        text_seg = model(img) + '~'
        
    text_seg = rep_this(text_seg,' ','')
            
    return text_seg


def mod_err(text_seg,imray):
    
    if text_seg.count(' ') > 0:
        text_seg = space_repl(text_seg,imray.copy())
    
    if '[he~' in text_seg:
        text_seg = the_repl(text_seg,imray.copy())
        
    idx = [[t.start(),t.end()] for t in re.finditer('\{([^}]+)', text_seg)]
    s = text_seg

    for i in idx:
        f = text_seg[i[0]:i[1]]
        find = [m.start() for m in re.finditer(r'\\', f)]
        for fn in find:
            fn = fn+i[0]
            if s[fn:fn+7] != '\\mathrm' and  s[fn+1] != '~':
                s = s[:fn] + '~' + s[fn + len('\\'):]
                
    text_seg = rep_this(s,'~~','~')
    
    return text_seg