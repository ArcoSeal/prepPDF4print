#!/usr/bin/env python3

import os
import argparse
import json
from operator import itemgetter

import numpy as np
from matplotlib import pyplot as plt
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.pdf import PageObject
from PIL import Image, ImageDraw
from pdf2image import convert_from_path

## Setup args & constants
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(THIS_FILE_DIR, './paper_sizes.json')) as f: PAPER_SIZES = json.load(f)
with open(os.path.join(THIS_FILE_DIR, './interior_margin_sizes.json')) as f: INTERIOR_MARGIN_SIZES = {int(kk): vv for kk, vv in json.load(f).items()}

def generate_document_coverage_array(pdf_path, page_range=None, sample_period=1, dpi=72, poppler_path=None):
    if page_range is None: page_range = (1, PdfFileReader(pdf_path).getNumPages())

    coverage_arr = None
    for pp in range(page_range[0], page_range[1]+1, sample_period):
        im = convert_from_path(pdf_path, first_page=pp, last_page=pp, dpi=dpi, poppler_path=poppler_path)[0]
        im_arr = np.array(im)
        if coverage_arr is None: coverage_arr = np.full(im_arr.shape[:2], False)
        coverage_arr += (im_arr != 255).all(axis=2)

    return coverage_arr

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

def find_nearest_paper_by_aspect(w, h, size_type='trim', subtract_margins=None):
    if size_type not in ('trim', 'bleed'): raise Exception()

    target_aspect_ratio = w / h

    paper_aspects = []
    for paper_sizes in PAPER_SIZES.values():
        paper_x, paper_y = paper_sizes[size_type]['x'], paper_sizes[size_type]['y']
        if subtract_margins:
            margin_l, margin_r, margin_t, margin_b = subtract_margins
            paper_x -= (margin_l + margin_r)
            paper_y -= (margin_t + margin_b)

        paper_aspects.append(paper_x / paper_y)

    paper_aspect_deltas = (abs(target_aspect_ratio - ii) for ii in paper_aspects)
    nearest_paper, aspect_delta = sorted(zip((paper_name for paper_name in PAPER_SIZES), paper_aspect_deltas), key=itemgetter(1))[0]
    return nearest_paper, aspect_delta

def find_margin_int_mm_from_pages(n_pages):
    for max_pages in sorted(INTERIOR_MARGIN_SIZES.keys()):
        if n_pages <= max_pages: return INTERIOR_MARGIN_SIZES[max_pages]
    raise Exception(f'Too many pages: {n_pages}. Specify margins manually.')

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

def annotate_found_margins(page_array, margins):
    margin_l, margin_r, margin_t, margin_b = margins

    im = Image.fromarray(page_array)
    im_annot = im.copy().convert('RGB')
    draw = ImageDraw.Draw(im_annot)

    margin_ul, margin_lr = (margin_l, margin_t), (im.size[0]-margin_r, im.size[1]-margin_b)
    draw.rectangle((margin_ul, margin_lr), outline='red')

    content_centroid_pil = midpoint(margin_lr, margin_ul, round_coords=True)
    draw = draw_cross(draw, content_centroid_pil, 5, 2, 'red')

    page_centroid_pil = midpoint((0,0), im.size, round_coords=True)
    draw = draw_cross(draw, page_centroid_pil, 5, 2, 'green')

    return im_annot

def user_adjust_margins(page_array, orginal_margins):
    margins = orginal_margins
    padding = [0, 0, 0, 0]

    redraw = True
    plt.ion()
    while True:
        if redraw:
            print(f'L/R/T/B: {margins}')
            page_im_annot = annotate_found_margins(page_array, margins)
            # page_im_annot.show()
            plt.imshow(page_im_annot)

        userinput = input('Adjust margins?: ').lower().strip()
        if userinput == '':
            plt.close()
            break

        elif len(userinput) < 3 or not (userinput[0] in 'lrtb' and userinput[1] in '-+' and userinput[2:].isdigit()):
            print('Invalid input, must be comma-separated groups of [lrtb][+-][0-9]+ (or "DONE" when done)')
            redraw = False

        else:
            plt.close()
            delta = int(userinput[2:]) * (-1 if userinput[1] == '-' else 1)
            padding['lrtb'.index(userinput[0])] += delta
            margins = tuple(sum(ii) for ii in zip(orginal_margins, padding))
            redraw = True

    plt.ioff()
    return margins, padding

def calculate_content_scale_factor(content_size_px, target_paper_size_mm, target_margin_int_mm, target_margin_ext_mm, target_margin_y_mm):
    target_content_size_mm = target_paper_size_mm[0]-target_margin_int_mm-target_margin_ext_mm, target_paper_size_mm[1]-2*target_margin_y_mm

    content_size_mm = content_size_px[0] / USU_PER_MM, content_size_px[1] / USU_PER_MM
    content_delta_x, content_delta_y = target_content_size_mm[0] - content_size_mm[0], target_content_size_mm[1] - content_size_mm[1]
    content_scale_x, content_scale_y = 1+content_delta_x/content_size_mm[0], 1+content_delta_y/content_size_mm[1]
    content_scale_factor = min(content_scale_x, content_scale_y)

    return content_scale_factor

def get_content_size_and_translation(source_page_mediabox, content_margins):
    content_ll = (source_page_mediabox.lowerLeft[0]+content_margins[0], source_page_mediabox.lowerLeft[1]+content_margins[3])
    content_ur = (source_page_mediabox.upperRight[0]-content_margins[1], source_page_mediabox.upperRight[1]-content_margins[2])
    content_size = (content_ur[0]-content_ll[0], content_ur[1]-content_ll[1])

    content_centroid = midpoint(content_ll, content_ur)
    page_centroid = midpoint((0,0), source_page_mediabox.upperRight)
    content_tx = round(page_centroid[0]-content_centroid[0]), round(page_centroid[1]-content_centroid[1]) # translation FROM content centroid TO page centroid

    return content_size, content_tx

def translate_and_scale_onto_target_paper(source_page, content_tx, content_scale_factor, target_paper_size_px):
    src_page_trans_scale = PageObject.createBlankPage(width=source_page.mediaBox.getWidth(), height=source_page.mediaBox.getHeight())
    src_page_trans_scale.mergeTranslatedPage(source_page, tx=content_tx[0], ty=content_tx[1])
    src_page_trans_scale.scaleBy(content_scale_factor)

    rescale_page_on_tgt_paper = PageObject.createBlankPage(width=target_paper_size_px[0], height=target_paper_size_px[1])
    rescale_page_on_tgt_paper.mergePage(src_page_trans_scale)

    return rescale_page_on_tgt_paper

def parse_cli_args():
    global USU_PER_MM

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input-pdf', required=True)
    argparser.add_argument('--output-pdf', default=None)
    argparser.add_argument('--paper-type', choices=PAPER_SIZES.keys(), default=None)
    argparser.add_argument('--margin-int', type=int, default=None, help='Interior margin (mm)')
    argparser.add_argument('--margin-ext', type=int, default=13, help='Exterior margin (mm)')
    argparser.add_argument('--margin-y', type=int, default=13, help='Vertical margin (mm)')
    argparser.add_argument('--no-rescale', action='store_true')
    argparser.add_argument('--dpi', type=int, default=72)
    argparser.add_argument('--coverage-sample-range', type=int, nargs=2, metavar=('FIRST_SAMPLE_PAGE', 'LAST_SAMPLE_PAGE'), default=None)
    argparser.add_argument('--coverage-sample-period', type=int, default=1)
    argparser.add_argument('--content-range', type=int, nargs=2, metavar=('FIRST_CONTENT_PAGE', 'LAST_CONTENT_PAGE'), default=None)
    argparser.add_argument('--poppler-path', help='Path to poppler/bin (if required)', default=None)

    args = argparser.parse_args()

    if not os.path.isfile(args.input_pdf): raise Exception(f'Not a file: {args.input_pdf}')

    USU_PER_MM = args.dpi / 25.4

    if args.content_range is None: args.content_range = (1, PdfFileReader(args.input_pdf).getNumPages())
    if args.coverage_sample_range is None: args.coverage_sample_range = args.content_range

    return args

if __name__ == '__main__':
    ## Setup
    args = parse_cli_args()
    n_pages = args.content_range[1]-args.content_range[0]+1
    pdf_reader = PdfFileReader(args.input_pdf)

    ## Figure out what margins we will use
    if args.margin_int is None:
        margin_int = find_margin_int_mm_from_pages(n_pages)
        print(f'Decided interior margin from page count ({n_pages}) -> {margin_int}mm')
    else:
        margin_int = args.margin_int
    margin_int, margin_ext, margin_y = find_margin_int_mm_from_pages(n_pages) if args.margin_int is None else args.margin_int, args.margin_ext, args.margin_y

    ## Figure out where & how big the content is
    coverage_arr = generate_document_coverage_array(args.input_pdf, args.coverage_sample_range, args.coverage_sample_period, args.dpi, poppler_path=args.poppler_path)
    content_margins = find_margins_px(coverage_arr)
    content_margins, content_margins_padding = user_adjust_margins(coverage_arr, content_margins)

    content_size, content_tx = get_content_size_and_translation(pdf_reader.getPage(args.content_range[0]-1).mediaBox, content_margins)

    ## Decide on target paper type
    if args.paper_type is None:
        target_paper_type = find_nearest_paper_by_aspect(*content_size, subtract_margins=(margin_int, margin_ext, margin_y, margin_y))[0]
        print(f'Closest paper type by aspect ratio: {target_paper_type}')
    else:
        target_paper_type = args.paper_type
        print(f'Use specified paper type: {target_paper_type}')

    target_paper_size_mm = PAPER_SIZES[target_paper_type]['trim']['x'], PAPER_SIZES[target_paper_type]['trim']['y']
    target_paper_size_px = (round(target_paper_size_mm[0]*USU_PER_MM), round(target_paper_size_mm[1]*USU_PER_MM))

    ## Scale & centre the content onto the target paper
    if not args.no_rescale:
        content_scale_factor = calculate_content_scale_factor(content_size, target_paper_size_mm, margin_int, margin_ext, margin_y)
    print(f'Content scale factor: {content_scale_factor}{" (overridden)" if args.no_rescale else ""}')

    transformed_pages = []
    for ii, page_num in enumerate(range(args.content_range[0], args.content_range[1]+1)):
        print(f'\rTransforming page {ii} ({ii/(args.content_range[1]+1-args.content_range[0])*100:.1f}%)...', end='')
        src_page = pdf_reader.getPage(page_num-1)
        rescale_page_on_tgt_paper = translate_and_scale_onto_target_paper(src_page, content_tx, content_scale_factor, target_paper_size_px)
        transformed_pages.append(rescale_page_on_tgt_paper)
    print('done')

    temp_transformed_path = './temp_transformed.pdf'
    pdf_out = PdfFileWriter()
    for ii in transformed_pages: pdf_out.addPage(ii)
    with open(temp_transformed_path, 'wb') as f: pdf_out.write(f)

    ## Check content location on target paper
    coverage_range_shifted = [ii-(args.content_range[0]-1) for ii in args.coverage_sample_range]
    transformed_coverage_arr = generate_document_coverage_array(temp_transformed_path, coverage_range_shifted, args.coverage_sample_period, args.dpi, poppler_path=args.poppler_path)
    content_margins_on_tgt_paper = find_margins_px(transformed_coverage_arr, pad=[round(ii*content_scale_factor) for ii in content_margins_padding])
    content_margins_on_tgt_paper, _ = user_adjust_margins(transformed_coverage_arr, content_margins_on_tgt_paper)

    os.remove(temp_transformed_path)

    ## Shift content to final location
    _, recentre_content_tx = get_content_size_and_translation(transformed_pages[0].mediaBox, content_margins_on_tgt_paper) # to recentre content on page
    margin_tx = (margin_int - (margin_int+margin_ext)/2) * USU_PER_MM # shift content according to interal/external margins (for odd page i.e. +ve if interior > exterior margin)

    pdf_out = PdfFileWriter()
    for ii, page_transformed in enumerate(transformed_pages):
        print(f'\rRecentering page {ii} ({ii/len(transformed_pages)*100:.1f}%)...', end='')
        margin_tx_thispage = margin_tx * (1 if (ii+1) % 2 else -1) # invert sign for even pages

        page_refit = PageObject.createBlankPage(width=target_paper_size_px[0], height=target_paper_size_px[1])
        page_refit.mergeTranslatedPage(page_transformed, tx=recentre_content_tx[0]+margin_tx_thispage, ty=recentre_content_tx[1])

        pdf_out.addPage(page_refit)
    print('done')

    output_path = os.path.splitext(args.input_pdf)[0]+f'_refit_{target_paper_type.replace("-", "_")}.pdf' if args.output_pdf is None else args.output_pdf

    with open(output_path, 'wb') as f: pdf_out.write(f)

    print(f'Wrote output to {output_path}')
