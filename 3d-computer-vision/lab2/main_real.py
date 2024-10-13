from utils import *
from plot_utils import *

def main():
    VIEW = '20160223_153830'
    PATH = f'./data/23_02_2016_stopalo/{VIEW}'
    SAVE_PATH = f'./data/zad2_imgs/{VIEW}'
    EXT  = 'jpg' 

    thresholds = find_illuminated_thresholds(path=PATH, ext=EXT)
    plot_img(thresholds)
    save_img(thresholds, SAVE_PATH, 'thresholds')

    sine_patterns = read_sine_patterns(path=PATH, ext=EXT)
    phases = calculate_phases()
    psi = calculate_wrapped_phase(sine_patterns, phases)
    plot_img(psi)
    save_img(psi, SAVE_PATH, 'wrapped_phase')

    imgs = read_gray_code_images(path=PATH, ext=EXT)
    gray_codes = calculate_gray_codes_fast(imgs, thresholds=thresholds)
    plot_img(gray_codes)
    save_img(gray_codes, SAVE_PATH, 'gray_codes')

    unwrapped = unwrap(psi, gray_codes)
    plot_img(unwrapped)
    save_img(unwrapped, SAVE_PATH, 'unwrapped')

main()