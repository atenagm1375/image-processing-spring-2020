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
    thresh_vals = threshold_multiotsu(im)
    regions = np.digitize(im, bins=thresh_vals)

    red_mask = np.zeros(regions.shape, np.uint8)
    white_mask = np.zeros(regions.shape, np.uint8)

    red_mask[regions == 1] = 255
    white_mask[regions == 0] = 255

    fig = plt.figure(figsize=(30, 30))
    fig.add_subplot(1, 4, 1)
    plt.imshow(im, 'gray')
    fig.add_subplot(1, 4, 2)
    plt.imshow(regions, 'gray')
    fig.add_subplot(1, 4, 3)
    plt.imshow(red_mask, 'gray')
    fig.add_subplot(1, 4, 4)
    plt.imshow(white_mask, 'gray')

    return red_mask, white_mask


def process_image(im):
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
    se_square_5 = np.ones((5, 5), np.uint8)

    close = cv2.morphologyEx(im, cv2.MORPH_CLOSE, se_circle_15, iterations=2)
    erode = cv2.erode(close, se_square_5, iterations=3)
    gradient = cv2.morphologyEx(erode, cv2.MORPH_GRADIENT, se_circle_15)

    cells = cv2.HoughCircles(gradient, cv2.HOUGH_GRADIENT, 1, 75,
                             param1=10, param2=18,
                             minRadius=15, maxRadius=120)

    hough = deepcopy(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))

    if cells is not None:
        cells = np.uint16(np.around(cells))
        # print(len(cells[0, :]))
        for i in cells[0, :]:
            cv2.circle(hough, (i[0], i[1]), i[2], (0, 255, 0), 8)

    fig = plt.figure(figsize=(30, 30))
    fig.add_subplot(1, 5, 1)
    plt.imshow(im, "gray", label="Thresholded image")
    fig.add_subplot(1, 5, 2)
    plt.imshow(close, "gray", label="Image after close operator")
    fig.add_subplot(1, 5, 3)
    plt.imshow(erode, "gray", label="Image after close and erode operators")
    fig.add_subplot(1, 5, 4)
    plt.imshow(gradient, "gray", label="Image gradients")
    fig.add_subplot(1, 5, 5)
    plt.imshow(hough, "gray", label="The cells")
    # fig.savefig("WHITE_{}.png".format(name))
    plt.show()
    plt.close()

    return cells

# %% COUNT THE CELLS


for ind, row in idb1_dataframe.iterrows():
    print("-*" * 50)
    print(f"Image {ind}:")
    red, white = distinguish_cells(row.image)
    print("Processing red...")
    red_cells = process_image(red)
    print("processing white...")
    white_cells = process_image(white)
    print(f"RED={len(red_cells[0, :])}, WHITE={len(white_cells[0, :])}")

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
