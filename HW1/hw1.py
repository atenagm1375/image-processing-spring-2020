import cv2
import numpy as np
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

    plt.imshow(mask * magnitude_spectrum, cmap='gray')
    plt.show()
    filtered_img = idft(mask * shifted_transformed_img)

    plot_before_after(img, filtered_img)


def image2():
    img = cv2.imread("2.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    # plot_before_after(img2, magnitude_spectrum, True)

    w, h = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)
    # TODO change these to a rectangular gaussian
    mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    filtered_img = cv2.blur(filtered_img, (2, 2))

    plot_before_after(img, filtered_img)


def image3():
    img = cv2.imread("3.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)

    w, h = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)
    # TODO change these to a gaussian notch
    mask[117 - 5:117 + 5, 95 - 5:95 + 5] = 0
    mask[178 - 5:178 + 5, 205 - 5:205 + 5] = 0

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    filtered_img = cv2.blur(filtered_img, (2, 2))

    plot_before_after(img, filtered_img)


def image4():
    pass


def image5():
    pass


def image6():
    pass


def image7():
    pass


def image8():
    pass


def image9():
    pass


def image10():
    pass


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
