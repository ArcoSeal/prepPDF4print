#!/usr/bin/env python3

import os
import argparse
import json
from operator import itemgetter

import numpy as np
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.pdf import PageObject
from PIL import Image, ImageDraw
from pdf2image import convert_from_path

## Setup args & constants
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

argparser = argparse.ArgumentParser()
argparser.add_argument('--input-pdf', required=True)
argparser.add_argument('--dpi', type=int, default=72)
argparser.add_argument('--front-cover', type=int, default=None)
argparser.add_argument('--back-cover', type=int, default=None)
argparser.add_argument('--content-range', type=int, nargs=2, default=None)
argparser.add_argument('--poppler-path', help='Path to poppler/bin', default=None)

args = argparser.parse_args()

INPUT_PDF = args.input_pdf

USU_PER_IN = args.dpi
IN_TO_MM = 25.4
USU_PER_MM = USU_PER_IN / IN_TO_MM

FRONT_COVER_PAGE = args.front_cover
BACK_COVER_PAGE = args.back_cover
CONTENT_RANGE = (1, PdfFileReader(INPUT_PDF).getNumPages()) if args.content_range is None else args.content_range
N_PAGES = CONTENT_RANGE[1]-CONTENT_RANGE[0]+1

POPPLER_PATH = args.poppler_path

with open(os.path.join(THIS_FILE_DIR, './paper_sizes.json')) as f: PAPER_SIZES = json.load(f)
with open(os.path.join(THIS_FILE_DIR, './gutter_sizes.json')) as f: GUTTER_SIZES = {int(kk): vv for kk, vv in json.load(f).items()}

def generate_document_coverage_array(pdf_path, page_range, sample_period=1, dpi=72, poppler_path=None):
    arr_overall = None
    for pp in range(*page_range, sample_period):
        print(f'\r{pp/page_range[1]*100:.0f}%...', end='')
        im = convert_from_path(pdf_path, dpi=dpi, first_page=pp, last_page=pp, poppler_path=poppler_path)[0]
        im_arr = np.array(im)
        if arr_overall is None: arr_overall = np.full(im_arr.shape[:2], False)
        arr_overall += (im_arr != 255).all(axis=2)
    print('done')

    return arr_overall

def find_margins_px(page_array, pad=(0,0,0,0)):
    pad_l, pad_r, pad_t, pad_b = pad

    cols_w_data = page_array.sum(axis=0).astype(bool)
    margin_l, margin_r = cols_w_data.argmax()-1 + pad_l, np.flip(cols_w_data).argmax()-1 + pad_r

    rows_w_data = page_array.sum(axis=1).astype(bool)
    margin_t, margin_b = rows_w_data.argmax()-1 + pad_t, np.flip(rows_w_data).argmax()-1 + pad_b

    return margin_l, margin_r, margin_t, margin_b

def get_page_size_mm(pdf_page):
    w_usu, h_usu = pdf_page.mediaBox.getWidth(), pdf_page.mediaBox.getHeight()
    w_mm, h_mm = w_usu / USU_PER_MM, h_usu / USU_PER_MM
    return w_mm, h_mm

def find_nearest_paper_by_aspect(w, h, size_type='trim', subtract_margins=None, subtract_gutter=None):
    if size_type not in ('trim', 'bleed'): raise Exception()

    target_aspect_ratio = w / h

    paper_aspects = []
    for paper_sizes in PAPER_SIZES.values():
        paper_x, paper_y = paper_sizes[size_type]['x'], paper_sizes[size_type]['y']
        if subtract_margins:
            margin_l, margin_r, margin_t, margin_b = subtract_margins
            paper_x -= (margin_l + margin_r)
            paper_y -= (margin_t + margin_b)

        if subtract_gutter:
            paper_x -= subtract_gutter

        paper_aspects.append(paper_x / paper_y)

    paper_aspect_deltas = (abs(target_aspect_ratio - ii) for ii in paper_aspects)
    nearest_paper, aspect_delta = sorted(zip((paper_name for paper_name in PAPER_SIZES), paper_aspect_deltas), key=itemgetter(1))[0]
    return nearest_paper, aspect_delta

def find_gutter_size_mm_from_pages(n_pages):
    for max_pages in sorted(GUTTER_SIZES.keys()):
        if n_pages <= max_pages: return GUTTER_SIZES[max_pages]
    raise Exception(f'Too many pages: {n_pages}. Is this actually a book?')

def midpoint(xy1, xy2, round_coords=False):
    x1, y1 = xy1
    x2, y2 = xy2

    midp = (x1+x2)/2, (y1+y2)/2
    if round_coords:
        return round(midp[0]), round(midp[1])
    else:
        return midp

def draw_cross(draw_obj, centre, length, thickness, colour):
    x, y = centre
    draw_obj.line(((x-length, y), (x+length, y)), colour, thickness)
    draw_obj.line(((x, y-length), (x, y+length)), colour, thickness)

    return draw_obj

def annotate_found_margins(page_array, margin_l, margin_r, margin_t, margin_b):
    im = Image.fromarray(page_array)
    im_annot = im.copy().convert('RGB')
    draw = ImageDraw.Draw(im_annot)

    margin_ul, margin_lr = (margin_l, margin_t), (im.size[0]-margin_r, im.size[1]-margin_b)
    draw.rectangle((margin_ul, margin_lr), outline='red')

    content_centroid_pil = midpoint(margin_lr, margin_ul, round_coords=True)
    draw = draw_cross(draw, content_centroid_pil, 5, 2, 'red')

    page_centroid_pil = midpoint((0,0), im.size, round_coords=True)
    draw = draw_cross(draw, page_centroid_pil, 5, 2, 'blue')

    return im_annot

TEST_PAGE = 26

# arr_overall = generate_document_coverage_array(INPUT_PDF, *CONTENT_RANGE, 10, USU_PER_IN, POPPLER_PATH)
arr_overall = generate_document_coverage_array(INPUT_PDF, [TEST_PAGE, TEST_PAGE+1], 10, USU_PER_IN, POPPLER_PATH)
margin_l, margin_r, margin_t, margin_b = find_margins_px(arr_overall)
# margin_l, margin_r, margin_t, margin_b = find_margins_px(arr_overall, pad_l=33)

print(f'L: {margin_l}, R: {margin_r}, T: {margin_t}, B: {margin_b}')

im_annot = annotate_found_margins(arr_overall, margin_l, margin_r, margin_t, margin_b)
im_annot.show()
print('asdsa')