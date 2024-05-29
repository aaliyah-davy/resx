from df2vid import *

# ALREADY RAN:
dpi = 400
inn = 'respdf/' + 'okay' + '.pdf'
imlist = get_rilines(inn,dpi)

indv_words, txt_list, full_text, sntc = get_imTeX(imlist)

import time
start = time.time()

imlist = ['grids/'+o for o in os.listdir('grids') if not o.startswith('._')]
imlist.sort()
imlist = [cv2.imread(i) for i in imlist]
section = abstract
sect_name = 'Abstract'
cite = title+'\n'+'Ci Fu, Aaliyah Davy, Simeon Holmes, Sheng Sun, Vikas Yadav, \
Asiya Gusa, Marco A. Coelho, Joseph Heitman' \
+'\n'+'PLoS Genet. 2021 Nov; 17(11): e1009935.'

video = get_video(section, sect_name, imlist, cite)
video.write_videofile("resx_vid.mp4",
                     codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True
                     )

end = time.time()
print(end - start)