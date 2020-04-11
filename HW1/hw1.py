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


def exe1():
    """
        Remove noise from images 1 to 4 with proper filters.
    """
    masked_fourier = shifted_transformed_img1 * mask
    filtered_img1 = idft(masked_fourier)
    mean_filter = (1 / 4) * np.ones((2, 2), dtype=np.uint8)
    filtered_img1 = cv2.filter2D(filtered_img1, -1, mean_filter)

    plot_before_after(img1, filtered_img1)

    img2 = cv2.imread("2.jpg", 0)
    shifted_transformed_img2, magnitude_spectrum = dft(img2)

    plot_before_after(img2, magnitude_spectrum, True)

    w, h = img2.shape
    mask = np.ones(img2.shape, dtype=np.uint8)
    mask[35 - 4:35 + 4, h // 2 - 4:h // 2 + 4] = 0
    mask[58 - 4:58 + 4, h // 2 - 4:h // 2 + 4] = 0
    mask[108 - 4:108 + 4, h // 2 - 4:h // 2 + 4] = 0
    mask[135 - 4:135 + 4, h // 2 - 4:h // 2 + 4] = 0
    mask[58 - 4:58 + 4, 10 - 4:10 + 4] = 0
    mask[108 - 4:108 + 4, 10 - 4:10 + 4] = 0
    mask[58 - 4:58 + 4, 212 - 4:212 + 4] = 0
    mask[108 - 4:108 + 4, 212 - 4:212 + 4] = 0
    mask[35 - 4:35 + 4, 10 - 4:10 + 4] = 0
    mask[135 - 4:135 + 4, 10 - 4:10 + 4] = 0
    mask[35 - 4:35 + 4, 212 - 4:212 + 4] = 0
    mask[135 - 4:135 + 4, 212 - 4:212 + 4] = 0

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img2 * mask
    filtered_img2 = idft(masked_fourier)
    mean_filter = (1 / 4) * np.ones((2, 2), dtype=np.uint8)
    filtered_img2 = cv2.filter2D(filtered_img2, -1, mean_filter)

    plot_before_after(img2, filtered_img2)


def exe2():
    """
        Improve the details and visibility of images 5 to 10 with spatial and
        frequency domain operations.
    """
    pass


def exe3():
    """
        Make images 11 and 12 younger with frequency domain operators.
    """
    pass


def __main__(num):
    name = "exe{}".format(num)
    globals()[name]()


if __name__ == "__main__":
    # try:
    __main__(int(sys.argv[1]))
    # except ValueError:
    #     print(
    #         "PLEASE ENTER QUESTION NUMBER AFTER PROGRAM NAME(A NUMBER IN RANGE [1, 3]).")
