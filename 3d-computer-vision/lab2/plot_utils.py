import matplotlib.pyplot as plt

def plot_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def plot_intensity_change(fi):
    r = fi[fi.shape[0]//2, :]
    plt.plot(list(range(r.shape[0])), r)
    plt.show()