#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

IMAGE PROCESSING HW2.

Author: Ashena Gorgan Mohammadi

"""


# %% IMPORT MODULES

# import sys
import os
import gc
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# %% LOAD DATA

ALL_IDB1_PATH_IM = "../../HW2/ALL_IDB1/im/"

idb1_images = {}
for file_name in os.listdir(ALL_IDB1_PATH_IM):
    idb1_images[file_name[:-6]] = {"image": cv2.imread(
            ALL_IDB1_PATH_IM + file_name), "class": file_name[-5]}

idb1_dataframe = pd.DataFrame.from_dict(idb1_images).transpose()

# %% EXTRACT GREEN CHANNEL AND NORMALIZE

idb1_dataframe["image"] = idb1_dataframe.image.apply(
        lambda im: cv2.split(im)[1])

idb1_dataframe["image"] = idb1_dataframe.image.apply(
        lambda im: cv2.convertScaleAbs(
                (im - np.min(im))*(255/(np.max(im) - np.min(im)))))

# %% PLOT ALL IMAGES

rows = int(np.ceil(len(idb1_dataframe.image) / 10))
i = 0
fig, axes = plt.subplots(nrows=rows, ncols=10, figsize=(30, 30))

for ind, x in idb1_dataframe.image.items():
    plt.title(ind)
    plt.axis("off")
    axes[i // 10, i % 10].imshow(x, "gray")
    i += 1
plt.show()

# %% GARBAGE COLLECTION

del i, rows, fig, axes, idb1_images
gc.collect()

# %% FUNCTIONS TO FIND THE CELLS


# def contrast_stretching(pixel, r1, s1, r2, s2):
#     if 0 <= pixel <= r1:
#         return pixel * (s1 / r1)
#     elif r1 < pixel <= r2:
#         return pixel * ((s2 - s1) / (r2 - r1)) + s1
#     else:
#         return pixel * ((255 - s2) / (255 - r2)) + s2


def distinguish_cells(im):
    """
    Distinguish red and white cells by applying dual otsu thresholding.

    Parameters
    ----------
    im : np.ndarray
        The image.

    Returns
    -------
    red_mask : np.ndarray
        Thresholded image containing only red cells.
    white_mask : np.ndarray
        Thresholded image containing only white cells.

    """
    blurred = cv2.GaussianBlur(im, (9, 9), 0.5)
    # gamma_corrected = cv2.equalizeHist(blurred)
    # gamma_corrected = cv2.createCLAHE(clipLimit=40).apply(gamma_corrected)
    # gamma_corrected = np.array(255 * (blurred / 255) ** 0.95, dtype=np.uint8)
    thresh_vals = threshold_multiotsu(blurred)
    regions = np.digitize(im, bins=thresh_vals)

    red_mask = np.zeros(regions.shape, np.uint8)
    white_mask = np.zeros(regions.shape, np.uint8)

    red_mask[regions == 1] = 255
    white_mask[regions == 0] = 255

    fig = plt.figure(figsize=(30, 30))
    fig.add_subplot(1, 3, 1)
    plt.imshow(im, 'gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(blurred, 'gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(regions, 'gray')

    return red_mask, white_mask


def process_image(im, name=None):
    """
    Process the thresholded image.

    Parameters
    ----------
    im : np.ndarray
        Thresholded image.

    Returns
    -------
    cells : np.ndarray
        The attributes of the cells in the image.

    """
    plt.imshow(im, 'gray')
    se_circle_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    se_cirlce_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opn = cv2.morphologyEx(im, cv2.MORPH_OPEN, se_cirlce_5, iterations=2)
    close = cv2.morphologyEx(opn, cv2.MORPH_CLOSE, se_circle_15, iterations=1)
    erode = cv2.erode(close, se_cirlce_5, iterations=1)
    gradient = cv2.morphologyEx(erode, cv2.MORPH_GRADIENT, se_circle_15)

    cells = cv2.HoughCircles(gradient, cv2.HOUGH_GRADIENT, 1, 50,
                             param1=50, param2=20,
                             minRadius=15, maxRadius=120)

    hough = deepcopy(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))

    if cells is not None:
        cells = np.uint16(np.around(cells))
        # print(len(cells[0, :]))
        for i in cells[0, :]:
            cv2.circle(hough, (i[0], i[1]), i[2], (255, 0, 0), 8)
    count = 0 if cells is None else len(cells[0, :])

    fig, axes = plt.subplots(1, 5)
    axes[0].imshow(im, "gray")
    # axes[0].set_title("Thresholded image")
    axes[1].imshow(close, "gray")
    # axes[1].set_title("Image after close operator")
    axes[2].imshow(erode, "gray")
    # axes[2].set_title("Image after close and erode operators")
    axes[3].imshow(gradient, "gray")
    # axes[3].set_title("Image gradients")
    axes[4].imshow(hough, "gray")
    axes[4].set_title(f"{count} cells")
    # fig.savefig("WHITE_{}.png".format(name))
    # plt.legend()
    plt.axis("off")
    plt.show()
    plt.close()

    if name is not None:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(hough, "gray")
        fig.savefig(name + ".png")
        plt.close()

    return cells

# %% COUNT THE CELLS


with open("log.txt", 'w') as file:
    for ind, row in idb1_dataframe.iterrows():
        file.write("-*" * 50 + "\n")
        file.write(f"Image {ind}:\n")
        print("-*" * 50)
        print(f"Image {ind}:")
        red, white = distinguish_cells(row.image)
        print("Processing red...")
        red_cells = process_image(red, f"{ind}_red")
        red_count = len(red_cells[0, :]) if red_cells is not None else 0
        print("processing white...")
        white_cells = process_image(white, f"{ind}_white")
        white_count = len(white_cells[0, :]) if white_cells is not None else 0
        file.write(f"RED={red_count}, WHITE={white_count}\n")
        print(f"RED={red_count}, WHITE={white_count}")

# %% COUNT WHITE CELLS


def count_white_cells(im):
    """
    Process image to find white cells.

    Parameters
    ----------
    im : np.ndarray
        The image.

    Returns
    -------
    None.

    """
    thresh_im = cv2.threshold(im, np.mean(im) // 2 - 1, 255,
                              cv2.THRESH_BINARY_INV)[1]
    thresh_im = cv2.medianBlur(thresh_im, 7)

    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opening_im = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, open_se)

    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closing_im = cv2.morphologyEx(opening_im, cv2.MORPH_CLOSE, close_se)

    erode_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    eroding_im = cv2.morphologyEx(closing_im, cv2.MORPH_ERODE, erode_se)

    grad_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    gradient = cv2.morphologyEx(eroding_im, cv2.MORPH_GRADIENT, grad_se)

    white_cells = cv2.HoughCircles(gradient, cv2.HOUGH_GRADIENT, 1, 75,
                                   param1=10, param2=18,
                                   minRadius=15, maxRadius=120)

    hough = deepcopy(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))

    if white_cells is not None:
        white_cells = np.uint16(np.around(white_cells))
        print(len(white_cells[0, :]))
        for i in white_cells[0, :]:
            cv2.circle(hough, (i[0], i[1]), i[2], (0, 255, 0), 8)

    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 6, 1)
    plt.imshow(im, "gray")
    fig.add_subplot(1, 6, 2)
    plt.imshow(thresh_im, "gray")
    fig.add_subplot(1, 6, 3)
    plt.imshow(opening_im, "gray")
    fig.add_subplot(1, 6, 4)
    plt.imshow(closing_im, "gray")
    fig.add_subplot(1, 6, 5)
    plt.imshow(gradient, "gray")
    fig.add_subplot(1, 6, 6)
    plt.imshow(hough, "gray")
#    fig.savefig("WHITE_{}.png".format(name))
    plt.show()
    plt.close()

    del open_se, close_se, erode_se, grad_se
    gc.collect()

    return (white_cells, thresh_im, opening_im, closing_im, eroding_im,
            gradient, hough)

# %% APPLY WHITE COUNT


white_output = idb1_dataframe['image'].apply(count_white_cells)

# %% COUNT RED CELLS


def count_red_cells(im):
    """
    Process image to find red cells.

    Parameters
    ----------
    im : np.ndarray
        The image.

    Returns
    -------
    None.

    """
    thresh_im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, 91, 3)
    thresh_im = cv2.medianBlur(thresh_im, 7)

    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    closing_im = cv2.morphologyEx(thresh_im, cv2.MORPH_CLOSE, close_se)

    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    opening_im = cv2.morphologyEx(closing_im, cv2.MORPH_OPEN, open_se)

    grad_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    gradient = cv2.morphologyEx(opening_im, cv2.MORPH_GRADIENT, grad_se)

    red_cells = cv2.HoughCircles(gradient, cv2.HOUGH_GRADIENT, 1, 75,
                                 param1=10, param2=18,
                                 minRadius=15, maxRadius=120)

    hough = deepcopy(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))

    whites = count_white_cells(im)
    white_cells = whites[0]

    red_cells = [red for red in red_cells if red not in white_cells]

    if red_cells is not None:
        red_cells = np.uint16(np.around(red_cells))
        print(len(red_cells[0, :]))
        for i in red_cells[0, :]:
            cv2.circle(hough, (i[0], i[1]), i[2], (0, 255, 0), 8)

    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 6, 1)
    plt.imshow(im, "gray")
    fig.add_subplot(1, 6, 2)
    plt.imshow(thresh_im, "gray")
    fig.add_subplot(1, 6, 3)
    plt.imshow(closing_im, "gray")
    fig.add_subplot(1, 6, 4)
    plt.imshow(opening_im, "gray")
    fig.add_subplot(1, 6, 5)
    plt.imshow(gradient, "gray")
    fig.add_subplot(1, 6, 6)
    plt.imshow(hough, "gray")
#    fig.savefig("WHITE_{}.png".format(name))
    plt.show()
    plt.close()

    del open_se, close_se, grad_se
    gc.collect()

    return (red_cells, thresh_im, opening_im, closing_im, gradient, hough)

# %%
