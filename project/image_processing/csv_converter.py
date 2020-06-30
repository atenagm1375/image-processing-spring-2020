
"""

IMAGE PROCESSING PROJECT CODE.

* CONCATENATE METADATA CSV FILES AND MAKE NEW CSV FILES.

"""

import os
import sys

import pandas as pd
import numpy as np


def concatenate_csv(src_folder_path, dest_file_path="metadata.csv"):
    """
    Load all metadata files and append them to make a single csv file.

    Parameters
    ----------
    src_folder_path : str
        Path of folder containing metadata files.
    dest_file_path : str, optional
        Destination to save the resulting file. The default is "metadata.csv".

    Returns
    -------
    df : pandas.DataFrame
        The final dataframe.

    """
    df = pd.DataFrame()

    for file in os.listdir(src_folder_path):
        try:
            df = df.append(pd.read_csv(src_folder_path + file))
        except FileNotFoundError:
            print("INVALID PATH")
            sys.exit(1)

    df.drop_duplicates(inplace=True)
    df.set_index("isic_id", inplace=True)

    try:
        df.to_csv(dest_file_path)
    except Exception:
        print("INVALID DESTINATION PATH")
        sys.exit(1)

    return df


def benign_malignant_dataframe(metadata_file_path, drop_na=True,
                               indeterminant=None, encoder=None,
                               save_to=None):
    """
    Generate a single dataframe to predict benign or malignant.

    Parameters
    ----------
    metadata_file_path : str
        Metadata file path.
    drop_na : bool, optional
        Remove rows with no label or not. The default is True.
    indeterminant : str, optional
        Whether to replace/remove values with indeterminant value or not.
        The valid strings are "remove_all" and "replace". The default is None.
    encoder : str, optional
        Whether to encode the data or not.
        The valid strings are "one_hot" and "label". The default is None.
    save_to : str, optional
        The path to save resulting file. The default is None.

    Raises
    ------
    ValueError
        If invalid options are given.

    Returns
    -------
    enc : pandas.DataFrame
        The resulting dataframe.

    """
    try:
        df = pd.read_csv(metadata_file_path, index_col=0)
    except FileNotFoundError:
        print("INVALID PATH")
        sys.exit(1)

    if indeterminant == "replace":
        df[df["benign_malignant"] == "indeterminate"] = np.nan
        df[df["benign_malignant"] == "indeterminate/benign"] = "benign"
        df[df["benign_malignant"] == "indeterminate/malignant"] = "malignant"
    elif indeterminant == "remove_all":
        df[df["benign_malignant"] == "indeterminate"] = np.nan
        df[df["benign_malignant"] == "indeterminate/benign"] = np.nan
        df[df["benign_malignant"] == "indeterminate/malignant"] = np.nan

    if drop_na:
        df.dropna(axis=0, subset=["benign_malignant"], inplace=True)
    else:
        df["benign_malignant"].fillna("indeterminate")

    if encoder is not None:
        if encoder.lower() in ["one_hot", "onehot"]:
            enc = pd.get_dummies(df["benign_malignant"])
        elif encoder.lower() == "label":
            labels = df["benign_malignant"].unique()
            enc = df["benign_malignant"].replace(labels)
        else:
            raise ValueError("INVALID ENCODER VALUE")
    else:
        enc = df["benign_malignant"]

    if save_to is not None:
        try:
            enc.to_csv(save_to)
        except Exception:
            print("INVALID DESTINATION PATH")
            sys.exit(1)

    return enc


def diagnosis_dataframe(metadata_file_path, drop_na=True,
                        encoder=None, save_to=None):
    """
    Generate a single dataframe to predict the diagnosis.

    Parameters
    ----------
    metadata_file_path : str
        Path to metadata file.
    drop_na : bool, optional
        Remove rows with no label or not. The default is True.
    encoder : str, optional
        Whether to encode the data or not.
        The valid strings are "one_hot" and "label". The default is None.
    save_to : str, optional
        The path to save resulting file. The default is None.

    Raises
    ------
    ValueError
        If invalid options are given.

    Returns
    -------
    enc : pandas.DataFrame
        The resulting dataframe.

    """
    try:
        df = pd.read_csv(metadata_file_path, index_col=0)
    except FileNotFoundError:
        print("INVALID PATH")
        sys.exit(1)

    if drop_na:
        df.dropna(axis=0, subset=["diagnosis"], inplace=True)
    else:
        df["diagnosis"].fillna("other")

    if encoder is not None:
        if encoder.lower() in ["one_hot", "onehot"]:
            enc = pd.get_dummies(df["diagnosis"])
        elif encoder.lower() == "label":
            labels = df["diagnosis"].unique()
            enc = df["diagnosis"].replace(labels)
        else:
            raise ValueError("INVALID ENCODER VALUE")
    else:
        enc = df["diagnosis"]

    if save_to is not None:
        try:
            enc.to_csv(save_to)
        except Exception:
            print("INVALID DESTINATION PATH")
            sys.exit(1)

    return enc
