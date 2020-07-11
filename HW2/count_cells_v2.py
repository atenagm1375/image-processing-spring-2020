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

# %% COUNT WHITE CELLS


def count_white_cells(im):
    thresh_im = cv2.threshold(im, np.mean(im) // 2 - 1, 255,
                              cv2.THRESH_BINARY_INV)[1]

    open_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opening_im = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, open_se)

    close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closing_im = cv2.morphologyEx(opening_im, cv2.MORPH_CLOSE, close_se)

    erode_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    eroding_im = cv2.morphologyEx(closing_im, cv2.MORPH_ERODE, erode_se)

    grad_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    gradient = cv2.morphologyEx(closing_im, cv2.MORPH_GRADIENT, grad_se)

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

# %%
