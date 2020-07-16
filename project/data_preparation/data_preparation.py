
"""

IMAGE PROCESSING PROJECT CODE.

* DATA SPLITING.

"""

import os
import gc
from shutil import move

import pandas as pd
import numpy as np


def split_data(metadata_file_path, n_parts=3, save_to=None):
    """
    Split data into (almost) equal parts.

    Parameters
    ----------
    metadata_file_path : str
        Path to metadata file.
    n_parts : int, optional
        Number of parts. The default is 3.
    save_to : str
        Path to folder where files are saved.

    Returns
    -------
    None.

    """
    df = pd.read_csv(metadata_file_path, index_col=0)
    inds = list(df.index)
    np.random.shuffle(inds)

    size, leftovers = divmod(len(inds), n_parts)
    chunks = []
    for i in range(n_parts):
        chunks.append(df.loc[inds[size * i: size * (i + 1)]])

    for i in range(leftovers):
        chunks[i % n_parts].append(df.iloc[size * n_parts + i])

    for i in range(n_parts):
        if save_to is None:
            chunks[i].to_csv(metadata_file_path[:-4] + "_{}.csv".format(i + 1))
        else:
            if not os.path.isdir(save_to):
                os.mkdir(save_to)
            chunks[i].to_csv(save_to + "metadata_{}.csv".format(i + 1))

    del chunks
    gc.collect()


def split_image_folders(metadata_files_paths, images_folder_path, folder):
    """
    Split images into separate folders.

    The folders are saved in a folder named DATA in the parent directory. Also
    moves the metadata files into this folder.

    Parameters
    ----------
    metadata_files_paths : list of str
        List of paths to metadata files related to each data section.
    images_folder_path : str
        Path to where the images reside.
    folder : str
        Folder to save images to.

    Returns
    -------
    None.

    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for i in range(len(metadata_files_paths)):
        df = pd.read_csv(metadata_files_paths[i], index_col=0)
        part_folder = folder + "part_{}/".format(i + 1)
        os.mkdir(part_folder)
        for file_name in df.index:
            try:
                move(images_folder_path + file_name + ".jpg",
                     folder + part_folder + file_name + ".jpg")
            except FileNotFoundError:
                print(f"part{i}, image: {file_name}")
                df.drop(file_name, axis=0, inplace=True)
        df.to_csv(metadata_files_paths[i])
        del df
        del part_folder
        gc.collect()
