# Res-X
### research paper pdf --> video (ai image + text-to-speech)

## Usage
Upload a PDF document and Res-X will generate a video. Works with papers that have LaTex-generated formulas.  (URL input options coming soon.)

It’s specifically made with researchers and students in mind due to the overwhelming expectation that they constantly stay up-to-date with new papers without enough time to read/parse everything. It may be particularly helpful for papers from an industry that the user is unfamiliar with, or for people who are visual learners/processors.

Unlike other platforms, Res-X is tailored to research papers and seeks to compartmentalize the input PDF and it works for papers with LaTex-generated formulas (like math/physics/compsci).

As a published researcher and visual learner, I definitely find Res-X useful.

### Example with my paper's abstract:
https://github.com/aaliyah-davy/resx/assets/134621145/9d574add-ad1d-415c-a970-ba422f8ad53b

Fu C, Davy A, Holmes S, Sun S, Yadav V, et al. (2021) Dynamic genome plasticity during unisexual reproduction in the human fungal pathogen Cryptococcus deneoformans. PLOS Genetics 17(11): e1009935. https://doi.org/10.1371/journal.pgen.1009935 


Res-X (Research Explanation) is a project that turns research papers into videos. It implements three types of Machine Learning models: Optical Character Recognition (Tesseract & LaTex OCR), Text-To-Speech, and Text-To-Image (StableDiffusion). 

## Requirements
scipy, torch, wkhtmltopdf, coqui-ai TTS, ffmpeg, moviepy, diffusers, os, requests, imgkit, io, sys, pdf2image, fake_useragent, re, cv2, math, string, imutils, numpy, regex, PIL, pandas, statistics, pytesseract, pix2tex, itertools, IPython, matplotlib, fontTools

## Reflection
Some of the major roadblocks I faced ultimately determined my approach:
- Started off web-scraping but could only scrape from some sites —> limited inputs to PDFs only
- PDF parsing libraries were inconsistent and often required perfectly formatted PDFs —> used document images and cv2 Contours library to parse
- cv2 Contours helped identify figures in doc image but not modern/small/medium tables —> acquiring table bounding boxes was time-intensive
- Text-To-Image required a GPU (which I don’t have) —> relied on Google Colab’s free TPU for that segment
