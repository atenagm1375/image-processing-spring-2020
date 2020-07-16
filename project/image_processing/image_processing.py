
"""

IMAGE PROCESSING PROJECT CODE.

* LOAD AND PROCESS DATA.

"""

import os
import gc
from copy import copy

import pandas as pd
import numpy as np
from PIL import Image, ImageFile


class Data:
    """
    Data class.

    Responsible for data preprocessing.
    """

    def __init__(self):
        self.images = []
        self.x = []
        self.y = []
        self.names = []

    def set_x(self, folder_path, image_size):
        """
        Set images(x).

        Parameters
        ----------
        folder_path : str
            Path to folder where the images reside.
        image_size : tuple of int
            The size to which images are resized.

        Returns
        -------
        None.

        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        print(folder_path)

        for file_name in os.listdir(folder_path):
            im = Image.open(folder_path + file_name).resize(image_size)
            if np.asarray(im).shape != (300, 300, 3):
                im = im.convert("RGB")
                print(file_name, np.asarray(im).shape)
            self.images.append(im)
            self.names.append(file_name[:-4])
            self.x.append(np.asarray(im))
        # print(self.x)
        # print(type(self.x))
        # self.x = np.array(self.x)
        gc.collect()

    def set_y(self, metadata_path):
        """
        Set data labels(y).

        Parameters
        ----------
        metadata_path : str
            Path to metadata file.

        Returns
        -------
        None.

        """
        df = pd.read_csv(metadata_path, index_col=0)
        names = copy(self.names)
        for name in names:
            if name in df.index:
                self.y.append(df.loc[name])
            else:
                ind = self.names.index(name)
                self.x.pop(ind)
                self.names.pop(ind)
        self.y = np.array(self.y)
        self.x = np.array(self.x)
        del df
        del names
        gc.collect()

    def powerlaw_transform(self, gamma, c):
        """
        Perform Powerlaw transformation(gamma correction) on x.

        Parameters
        ----------
        gamma : float
            Image enhancement parameter.
        c : int
            The bit size.

        Returns
        -------
        None.

        """
        self.x = np.array(list(map(lambda r: c * self.x ** gamma, self.x)))
