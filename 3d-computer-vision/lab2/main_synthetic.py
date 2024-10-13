from utils import *
from plot_utils import *

def main():
    SAVE_PATH = './data/zad1_imgs'

    sine_patterns = read_sine_patterns()
    phases = calculate_phases()
    psi = calculate_wrapped_phase(sine_patterns, phases)
    plot_img(psi)
    plot_intensity_change(psi)
    save_img(psi, SAVE_PATH, 'wrapped_phase')
    save_intensity_change(psi, SAVE_PATH, 'wrapped_phase_intensity')

    imgs = read_gray_code_images()
    gray_codes = calculate_gray_codes_fast(imgs)
    plot_img(gray_codes)
    plot_intensity_change(gray_codes)
    save_img(gray_codes, SAVE_PATH, 'gray_codes')
    save_intensity_change(gray_codes, SAVE_PATH, 'gray_codes_intensity')

    unwrapped = unwrap(psi, gray_codes)
    plot_img(unwrapped)
    plot_intensity_change(unwrapped)
    save_img(unwrapped, SAVE_PATH, 'unwrapped_phase')
    save_intensity_change(unwrapped, SAVE_PATH, 'unwrapped_phase_intensity')

main()