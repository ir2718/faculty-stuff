import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from plot_utils import *
from scipy.ndimage import median_filter
from tqdm import tqdm
from sympy.combinatorics.graycode import gray_to_bin

def read_sine_patterns(path='./data/640x480 GC+PS (cols) - LeftRightInverted', ext='png', coords=None):
    imgs = []
    for i in tqdm(range(1, 9)):
        img = Image.open(f'{path}/slika{i}.{ext}').convert('L')
        img_np = np.asarray(img)/255.
        imgs.append(img_np)
    return np.array(imgs)

def calculate_phases(N=8):
    return np.array([(2 * np.pi * i) / N for i in range(N)])

def calculate_wrapped_phase(sine_patterns, phases):
    y = np.sum( sine_patterns * np.sin(phases).reshape(-1, 1, 1), axis=0) ### nema minusa?
    x = np.sum( sine_patterns * np.cos(phases).reshape(-1, 1, 1), axis=0)
    fi = np.arctan2(y, x)

    fi = (fi - np.pi/2) % (2 * np.pi) ### BITNO
    return  fi
    
def read_gray_code_images(path='./data/640x480 GC+PS (cols) - LeftRightInverted', ext='png', range_=(9, 13)):
    imgs = []
    for i in tqdm(range(range_[0], range_[1])):
        img = Image.open(f'{path}/slika{i}.{ext}').convert('L')
        img_np = np.asarray(img)/255.
        imgs.append(img_np)
    #imgs[0] = np.flip(imgs[0], axis=1)
    return np.array(imgs)

def calculate_gray_codes(imgs, thresholds=None):
    gray_code = np.zeros(shape=(imgs.shape[1:]))
    for i in tqdm(range(imgs.shape[1])):
        for j in range(imgs.shape[2]):
            code = imgs[:, i, j]
            if thresholds is not None:
                code = (code >= thresholds[i, j])
            code = ''.join(code.astype(np.int32).astype(str))

            b = gray_to_bin(code)
            v = int(b, 2)
            gray_code[i, j] = v

    #gray_code = np.sort(gray_code, axis=1)
    #gray_code = np.flip(gray_code, axis=1)
    return gray_code

def calculate_gray_codes_fast(imgs, thresholds=None):
    mapping = {
        '0000': 0, '0001': 1, '0011': 2, '0010': 3, '0110': 4, '0111': 5, '0101': 6, 
        '0100': 7, '1100': 8, '1101': 9, '1111': 10, '1110': 11, '1010': 12, '1011': 13,
        '1001': 14, '1000': 15 
    }

    gray_code = np.zeros(shape=(imgs.shape[1:]))
    for i in tqdm(range(imgs.shape[1])):
        for j in range(imgs.shape[2]):
            code = imgs[:, i, j]
            if thresholds is not None:
                code = (code >= thresholds[i, j])
            code = ''.join(code.astype(np.int32).astype(str))
            gray_code[i, j] = mapping[code]

    return gray_code

def unwrap(psi, gray_codes):
    unwrapped = psi + gray_codes * (2 * np.pi)
    unwrapped01 = (unwrapped - unwrapped.min())/(unwrapped.max() - unwrapped.min())
    return unwrapped01

def read_background(path, ext):
    return np.asarray(Image.open(f'{path}/slika18.{ext}').convert('L'))/255., \
         np.asarray(Image.open(f'{path}/slika17.{ext}').convert('L'))/255.

def find_illuminated_thresholds(path, ext):
    illuminated = np.asarray(Image.open(f'{path}/slika18.{ext}').convert('L'))
    not_illuminated = np.asarray(Image.open(f'{path}/slika17.{ext}').convert('L'))
    return ((illuminated + not_illuminated)/2.).reshape(illuminated.shape) / 255.

def save_img(img, path, name):
    plt.imsave(f'{path}/{name}.png', img, cmap='gray')

def save_intensity_change(img, path, name):
    r = img[img.shape[0]//2, :]
    plt.plot(list(range(r.shape[0])), r)
    plt.savefig(f'{path}/{name}.png')
    plt.clf()