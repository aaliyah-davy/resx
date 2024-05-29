# Res-X
## research paper pdf --> video (ai image + text-to-speech)

Res-X (Research Explanation) is a project that turns research papers into videos. It implements three types of Machine Learning models: Optical Character Recognition (Tesseract & LaTex OCR), Text-To-Speech, and Text-To-Image (StableDiffusion). 
Libraries: scipy, gets, torch, TTS, ffmpeg, movie-you.editor, diffusers, os, requests, imgkit, io, sys, pdf2image, fake_useragent, re, cv2, math, string, imutils, numpy, regex, PIL, pandas, statistics, pytesseract, pix2tex, itertools, IPython, matplotlib

Upload a PDF document and Res-X will generate a video. Works with papers that have LaTex-generated formulas.  (URL input options coming soon.)

It’s specifically made with researchers and students in mind due to the overwhelming expectation that they constantly stay up-to-date with new papers without enough time to read/parse everything. It may be particularly helpful for papers from an industry that the user is unfamiliar with, or for people who are visual learners/processors.

Unlike other platforms, Res-X is tailored to research papers and seeks to compartmentalize the input PDF and it works for papers with LaTex-generated formulas (like math/physics/compsci).

As a published researcher and visual learner, I definitely find Res-X useful.

Some of the major roadblocks I faced ultimately determined my approach:
- Started off web-scraping but could only scrape from some sites —> limited inputs to PDFs only
- PDF parsing libraries were inconsistent and often required perfectly formatted PDFs —> used document images and cv2 Contours library to parse
- cv2 Contours helped identify figures in doc image but not modern/small/medium tables —> acquiring table bounding boxes was time-intensive
- Text-To-Image required a GPU (which I don’t have) —> relied on Google Colab’s free TPU for that segment



https://github.com/aaliyah-davy/resx/assets/134621145/02972449-e0da-4da2-a218-4873944d6854

https://github.com/aaliyah-davy/resx/assets/134621145/53c84631-34e1-4032-a8b1-2844d79a3c2d


