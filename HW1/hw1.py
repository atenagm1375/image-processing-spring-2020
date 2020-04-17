import cv2
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import sys


def plot_before_after(img1, img2, is_magnitude=False):
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    if is_magnitude:
        plt.title('Magnitude Spectrum')
    else:
        plt.title('Filtered Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def dft(img):
    transformed_img = np.fft.fft2(img)
    shifted_transformed_img = np.fft.fftshift(transformed_img)
    magnitude_spectrum = np.log(1 + np.abs(shifted_transformed_img))
    return shifted_transformed_img, magnitude_spectrum


def idft(img):
    filtered_img = np.fft.ifftshift(img)
    filtered_img = np.fft.ifft2(
        filtered_img).real.clip(0, 255).astype(np.uint8)
    return filtered_img


def gaussian_lowpass_filter(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gaussian_band_reject(shape=(3, 3), sigma=0.5, w=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = 1 - np.exp(-(((x * x + y * y) - sigma**2) /
                     (np.sqrt(x * x + y * y) * w))**2)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def homomorphic_filtering(img, rh, rl, cutoff, c=1):
    img = np.float32(img)
    img = img / 255
    rows, cols = img.shape

    img_log = np.log(img + 1)

    img_fft_shift, _ = dft(img_log)

    DX = cols / cutoff
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j] = ((rh - rl) * (1 - np.exp(-c * ((i - rows / 2) **
                                                     2 + (j - cols / 2)**2) / (2 * DX**2)))) + rl

    result_filter = G * img_fft_shift

    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

    return np.exp(result_interm)


def make_younger(img, sigma):
    transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)
    h = cv2.normalize(gaussian_lowpass_filter(img.shape, sigma=sigma),
                      None, 0, 1, cv2.NORM_MINMAX)
    transformed_img = transformed_img * h
    filtered_img = idft(transformed_img)
    return filtered_img


def image1():
    img = cv2.imread('1.jpg', 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    plot_before_after(img, magnitude_spectrum, True)

    # mask = cv2.normalize(gaussian_band_reject(
    #     img.shape, 88, 9), None, 0, 1, cv2.NORM_MINMAX)
    mask = cv2.normalize(gaussian_lowpass_filter(
        img.shape, 17), None, 0, 1, cv2.NORM_MINMAX)
    # mask = np.ones(img.shape, dtype=np.uint8)
    # w, h = img.shape
    # mask = np.ones(img.shape, dtype=np.uint8)
    # # TODO change these to a rectangular gaussian
    # mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    # mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0
    # mask[w // 2 - 2:w // 2 + 2, 0:h // 2 - 5] = 0
    # mask[w // 2 - 2:w // 2 + 2, h // 2 + 5:h] = 0

    plt.imshow(mask * magnitude_spectrum, cmap='gray')
    plt.show()
    filtered_img = idft(mask * shifted_transformed_img)

    plot_before_after(img, filtered_img)


def image2():
    img = cv2.imread("2.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    # plot_before_after(img2, magnitude_spectrum, True)

    w, h = img.shape
    mask = np.ones(img.shape)
    # TODO change these to a rectangular gaussian
    mask[0:w // 2 - 5, h // 2] = 0
    mask[0:w // 2 - 5, h // 2 + 1] = 0.3
    mask[0:w // 2 - 5, h // 2 + 2] = 0.7
    mask[0:w // 2 - 5, h // 2 - 2] = 0.7
    mask[0:w // 2 - 5, h // 2 - 1] = 0.3
    mask[w // 2 + 5:w, h // 2] = 0
    mask[w // 2 + 5:w, h // 2 + 1] = 0.3
    mask[w // 2 + 5:w, h // 2 + 2] = 0.7
    mask[w // 2 + 5:w, h // 2 - 2] = 0.7
    mask[w // 2 + 5:w, h // 2 - 1] = 0.3

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    filtered_img = cv2.blur(filtered_img, (2, 2))

    plot_before_after(img, filtered_img)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def image3():
    img = cv2.imread("3.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    mask = np.ones(img.shape)
    mask[117 - 5:117 + 5, 96 - 5:96 + 5] = gkern(10, 1)
    mask[85 - 5:85 + 5, 42 - 5:42 + 5] = gkern(10, 1)
    mask[178 - 5:178 + 5, 204 - 5:204 + 5] = gkern(10, 1)
    mask[210 - 5:210 + 5, 258 - 5:258 + 5] = gkern(10, 1)
    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()
    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    plot_before_after(img, filtered_img)


def image4():
    img = cv2.imread("4.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)


def image5():
    img = cv2.imread("5.jpg", 0)
    # filtered_img = np.array(255 * (img / 255)**1.5, dtype=np.uint8)
    # filtered_img = homomorphic_filtering(img, 1, 1.5, 100)
    filtered_img = cv2.equalizeHist(img)
    # filtered_img = cv2.medianBlur(filtered_img, 3)
    # filtered_img = cv2.GaussianBlur(filtered_img, (3, 3), 0)
    # filtered_img = cv2.blur(filtered_img, (5, 5))
    plot_before_after(img, filtered_img)


def image6():
    img = cv2.imread("6.jpg", 0)
    # filtered_img = np.array(255 * (img / 255)**.3, dtype=np.uint8)
    filtered_img = homomorphic_filtering(img, 0.1, 0.7, 30, 0.1)
    # filtered_img = cv2.equalizeHist(img)
    plot_before_after(img, filtered_img)
    plot_before_after(img, filtered_img - img)


def image7():
    img = cv2.imread("7.jpg", 0)
    # filtered_img = cv2.equalizeHist(img)
    filtered_img = homomorphic_filtering(img, 0.8, 0.7, 20, 50)
    plot_before_after(img, filtered_img)


def image8():
    img = cv2.imread("8.jpg", 0)
    filtered_img = homomorphic_filtering(img, 0.8, 0.7, 20, 50)
    plot_before_after(img, filtered_img)


def image9():
    img = cv2.imread("9.jpg", 0)
    filtered_img = homomorphic_filtering(img, 0.8, 0.7, 20, 50)
    plot_before_after(img, filtered_img)


def image10():
    img = cv2.imread("10.jpg", 0)
    filtered_img = homomorphic_filtering(img, 0.9, 0.75, 10, 100)
    plot_before_after(img, filtered_img)


def image11():
    img = cv2.imread("11.jpg", 0)
    younger_img = make_younger(img, 40)
    plot_before_after(img, younger_img)


def image12():
    img = cv2.imread("12.jpg", 0)
    younger_img = make_younger(img, 32)
    plot_before_after(img, younger_img)


def __main__(num):
    name = "image{}".format(num)
    globals()[name]()


if __name__ == "__main__":
    # try:
    __main__(int(sys.argv[1]))
    # except ValueError:
    #     print(
    #         "PLEASE ENTER QUESTION NUMBER AFTER PROGRAM NAME(A NUMBER IN RANGE [1, 3]).")
