
"""

IMAGE PROCESSING PROJECT CODE.

* LOAD AND PROCESS DATA.

"""

import os
import gc

import pandas as pd
import numpy as np
from PIL import Image


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
        for file_name in os.listdir(folder_path):
            self.images.append(Image.open(folder_path + file_name).resize(
                image_size))
            self.names.append(file_name[:-4])
        self.x = np.array(self.images)
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
        df = pd.load_csv(metadata_path, index_col=0)
        for name in self.names:
            self.y.append(df[name])
        self.y = np.array(self.y)
        del df
        gc.collect()

    def powerlaw_transformation(self, gamma, c):
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
