"""
IMAGE PROCESSING HW1.

ASHENA G.MOHAMMADI
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys


def plot_before_after(img1, img2, is_magnitude=False):
    """
    Plot images before and after editing alongside.

    Parameters
    ----------
    img1 : numpy.array
        The original image.
    img2 : numpy.array
        The editted image.
    is_magnitude : bool, optional
        Whether to plot the magnitude spectrum or not. The default is False.

    Returns
    -------
    None.

    """
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
    """
    Transform the image into frequency domain.

    Parameters
    ----------
    img : numpy.array
        The original image.

    Returns
    -------
    shifted_transformed_img : numpy.array
        The transformed image.
    magnitude_spectrum : numpy.array
        Magnitude spectrum of the image.

    """
    transformed_img = np.fft.fft2(img)
    shifted_transformed_img = np.fft.fftshift(transformed_img)
    magnitude_spectrum = np.log(1 + np.abs(shifted_transformed_img))
    return shifted_transformed_img, magnitude_spectrum


def idft(img):
    """
    Transform image from frequency domain into spatial domain.

    Parameters
    ----------
    img : numpy.array
        An image in frequency domain.

    Returns
    -------
    filtered_img : numpy.array
        The image in spatial domain.

    """
    filtered_img = np.fft.ifftshift(img)
    filtered_img = np.fft.ifft2(
        filtered_img).real.clip(0, 255).astype(np.uint8)
    return filtered_img


def gaussian_lowpass_filter(shape=(3, 3), sigma=0.5):
    """
    Create a lowpass Gaussian filter.

    Parameters
    ----------
    shape : tuple, optional
        Size of the filter. The default is (3, 3).
    sigma : float, optional
        The Gaussian parameter sigma. The default is 0.5.

    Returns
    -------
    h : numpy.array
        The filter.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gaussian_band_reject(shape=(3, 3), sigma=0.5, w=1):
    """
    Create a band reject Gaussian filter.

    Parameters
    ----------
    shape : tuple, optional
        Size of the filter. The default is (3, 3).
    sigma : float, optional
        The Gaussian parameter sigma. The default is 0.5.
    w : float, optional
        The bandwidth. The default is 1.

    Returns
    -------
    h : numpy.array
        The filter.

    """
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
    """
    Apply homomorphic filter.

    Parameters
    ----------
    img : numpy.array
        The original image.
    rh : float
        Gamma high parameter.
    rl : float
        Gamma low parameter.
    cutoff : float
        Cutoff value.
    c : float, optional
        The multiplier c in the filter. The default is 1.

    Returns
    -------
    numpy.array
        The filtered image.

    """
    img = np.float32(img)
    img = img / 255
    rows, cols = img.shape

    img_log = np.log(img + 1)

    img_fft_shift, _ = dft(img_log)

    DX = cols / cutoff
    G = np.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            G[i][j] = ((rh - rl) * (1 - np.exp(-c * ((i - rows / 2) ** 2 +
                                                     (j - cols / 2)**2) /
                                               (2 * DX**2)))) + rl

    result_filter = G * img_fft_shift

    result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

    return np.exp(result_interm)


def make_younger(img, sigma):
    """
    Reduce wrinkles from faces to make them look younger.

    Parameters
    ----------
    img : numpy.array
        The original image.
    sigma : float
        Parameter sigma in a Gaussian lowpass filter.

    Returns
    -------
    filtered_img : numpy.array
        The filtered image.

    """
    transformed_img, magnitude_spectrum = dft(img)
    h = cv2.normalize(gaussian_lowpass_filter(img.shape, sigma=sigma),
                      None, 0, 1, cv2.NORM_MINMAX)
    transformed_img = transformed_img * h
    filtered_img = idft(transformed_img)
    return filtered_img


def image1():
    """
    Edit image 1.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread('1.jpg', 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    plot_before_after(img, magnitude_spectrum, True)

    w, h = img.shape
    mask = np.ones(img.shape, dtype=np.uint8)

    for i in range(w):
        if np.mean(magnitude_spectrum[i, :]) >= 9.:
            # magnitude_spectrum[i, :] = 0
            mask[i, :] = 0

    for j in range(h):
        if np.mean(magnitude_spectrum[:, j]) >= 9.:
            # magnitude_spectrum[:, j] = 0
            mask[:, j] = 0
    mask[w // 2 - 5:w // 2 + 5, h // 2 - 5:h // 2 + 5] = 1

    plt.imshow(mask * magnitude_spectrum, cmap='gray')
    plt.show()

    filtered_img = idft(mask * shifted_transformed_img)
    filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)

    plot_before_after(img, filtered_img)


def image2():
    """
    Edit image 2.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("2.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)

    w, h = img.shape
    mask = np.ones(img.shape)
    mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()

    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    filtered_img = cv2.blur(filtered_img, (2, 2))

    plot_before_after(img, filtered_img)


def image3():
    """
    Edit image 3.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("3.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    w, h = img.shape
    mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    cv2.circle(mask, (96, 117), 10, (0, 0, 0), -1)
    cv2.circle(mask, (42, 85), 10, (0, 0, 0), -1)
    cv2.circle(mask, (204, 178), 10, (0, 0, 0), -1)
    cv2.circle(mask, (258, 210), 10, (0, 0, 0), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    plt.imshow(magnitude_spectrum * mask, cmap='gray')
    plt.show()
    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    plot_before_after(img, filtered_img)


def image4():
    """
    Edit image 4.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("4.jpg", 0)
    shifted_transformed_img, magnitude_spectrum = dft(img)
    plot_before_after(img, magnitude_spectrum, True)

    w, h = img.shape
    # mask = np.ones(img.shape)
    # # for i in range(w):
    # #     for j in range(h):
    # #         if 2 * magnitude_spectrum[i, j] > 25:
    # #             mask[i, j] = 0
    # for i in range(w):
    #     if np.mean(magnitude_spectrum[i, :]) >= 9.:
    #         # magnitude_spectrum[i, :] = 0
    #         mask[i, :] = 0
    #
    # for j in range(h):
    #     if np.mean(magnitude_spectrum[:, j]) >= 9.:
    #         # magnitude_spectrum[:, j] = 0
    #         mask[:, j] = 0
    # mask[w // 2 - 5:w // 2 + 5, h // 2 - 5:h // 2 + 5] = 1
    # mask = cv2.normalize(gaussian_lowpass_filter(
    #     img.shape, 40), None, 0, 1, cv2.NORM_MINMAX)
    mask = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # cv2.line(mask, (428, 134), (245, 280), (0, 0, 0), 8)
    # cv2.line(mask, (509, 232), (252, 435), (0, 0, 0), 8)
    # cv2.line(mask, (493, 142), (370, 260), (0, 0, 0), 8)
    # cv2.line(mask, (345, 280), (213, 396), (0, 0, 0), 8)
    cv2.line(mask, (155, 150), (155, 350), (0, 0, 0), 8)
    cv2.line(mask, (205, 150), (205, 370), (0, 0, 0), 8)
    cv2.line(mask, (240, 110), (240, 400), (0, 0, 0), 8)
    cv2.line(mask, (0, 265), (350, 265), (0, 0, 0), 8)
    cv2.line(mask, (375, 265), (h, 265), (0, 0, 0), 8)
    cv2.line(mask, (290, 110), (290, 400), (0, 0, 0), 8)
    cv2.line(mask, (410, 110), (410, 400), (0, 0, 0), 8)
    cv2.line(mask, (510, 110), (510, 400), (0, 0, 0), 8)
    cv2.line(mask, (345, 70), (150, 255), (0, 0, 0), 8)
    cv2.line(mask, (370, 505), (590, 270), (0, 0, 0), 8)
    cv2.line(mask, (445, 90), (445, 460), (0, 0, 0), 8)
    cv2.line(mask, (540, 150), (540, 350), (0, 0, 0), 8)
    cv2.line(mask, (0, 190), (h, 190), (0, 0, 0), 8)
    cv2.line(mask, (0, 340), (h, 340), (0, 0, 0), 8)
    cv2.line(mask, (0, 305), (h, 305), (0, 0, 0), 8)
    cv2.line(mask, (0, 220), (h, 220), (0, 0, 0), 8)
    cv2.circle(mask, (270, 440), 10, (0, 0, 0), -1)
    cv2.circle(mask, (490, 160), 10, (0, 0, 0), -1)
    cv2.rectangle(mask, (0, 0), (h, 70), (0, 0, 0), -1)
    cv2.rectangle(mask, (0, 480), (h, w), (0, 0, 0), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[0:w // 2 - 5, h // 2 - 2:h // 2 + 2] = 0
    mask[w // 2 + 5:w, h // 2 - 2:h // 2 + 2] = 0

    plt.imshow(20 * magnitude_spectrum * mask, cmap='gray')
    plt.show()
    masked_fourier = shifted_transformed_img * mask
    filtered_img = idft(masked_fourier)
    plot_before_after(img, filtered_img)


def image5():
    """
    Edit image 5.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("5.jpg", 0)
    filtered_img = np.array(255 * (img / 255)**1.5, dtype=np.uint8)
    filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0.1)
    plot_before_after(img, filtered_img)


def image6():
    """
    Edit image 6.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("6.jpg", 0)
    # filtered_img = np.array(255 * (img / 255)**0.9, dtype=np.uint8)
    # filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)
    # kernel = np.array([
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0]
    # ])
    # filtered_img = cv2.filter2D(img, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(3, 3))
    filtered_img = clahe.apply(img)
    blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 1)
    # mask = img - blurred_img
    # plot_before_after(img, mask)
    # filtered_img = img + 0.1 * mask
    filtered_img = cv2.addWeighted(filtered_img, 2, blurred_img, -1, -2)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)
    # filtered_img = homomorphic_filtering(img, 0.8, 0.6, 100, 2)
    # filtered_img = cv2.equalizeHist(img)
    plot_before_after(img, np.clip(closing, 0, 255))
    # plot_before_after(img, filtered_img - img)


def image7():
    """
    Edit image 7.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("7.jpg", 0)
    # filtered_img = np.array(255 * (img / 255)**0.5, dtype=np.uint8)
    # # filtered_img = cv2.GaussianBlur(filtered_img, (5, 5), 0)
    # blurred_img = cv2.GaussianBlur(filtered_img, (7, 7), 5)
    # mask = img - blurred_img
    # plot_before_after(img, mask)
    # filtered_img = img + 0.2 * mask
    # filtered_img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    filtered_img = clahe.apply(img)
    # blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 0.2)
    # mask = img - blurred_img
    # plot_before_after(img, mask)
    # filtered_img = img + 0.1 * mask
    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel)
    filtered_img = homomorphic_filtering(filtered_img, 0.9, 0.7, 20, 2)
    plot_before_after(img, filtered_img)


def image8():
    """
    Edit image 8.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("8.jpg", 0)
    filtered_img = np.array(255 * (img / 255)**0.7, dtype=np.uint8)
    # filtered_img = cv2.medianBlur(filtered_img, 3)
    blurred_img = cv2.GaussianBlur(filtered_img, (5, 5), 0.01)
    mask = img - blurred_img
    plot_before_after(img, mask)
    filtered_img = img + 0.08 * mask
    plot_before_after(img, filtered_img)


def image9():
    """
    Edit image 9.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("9.jpg", 0)
    filtered_img = np.array(255 * (img / 255)**0.9, dtype=np.uint8)
    blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 3)
    mask = img - blurred_img
    plot_before_after(img, mask)
    filtered_img = img + 0.1 * mask
    plot_before_after(img, filtered_img)


def image10():
    """
    Edit image 10.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("10.jpg", 0)
    filtered_img = np.array(255 * (img / 255)**0.5, dtype=np.uint8)
    blurred_img = cv2.GaussianBlur(filtered_img, (3, 3), 1.5)
    mask = img - blurred_img
    plot_before_after(img, mask)
    filtered_img = img + 0.1 * mask
    plot_before_after(img, filtered_img)


def image11():
    """
    Edit image 11.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("11.jpg", 0)
    younger_img = make_younger(img, 40)
    plot_before_after(img, younger_img)


def image12():
    """
    Edit image 12.jpg.

    Returns
    -------
    None.

    """
    img = cv2.imread("12.jpg", 0)
    younger_img = make_younger(img, 40)
    plot_before_after(img, younger_img)


def __main__(num):
    name = "image{}".format(num)
    globals()[name]()


if __name__ == "__main__":
    try:
        __main__(int(sys.argv[1]))
    except ValueError:
        print(
            "PLEASE ENTER IMAGE NUMBER AFTER PROGRAM NAME(A NUMBER IN",
            "RANGE [1, 12]).")
