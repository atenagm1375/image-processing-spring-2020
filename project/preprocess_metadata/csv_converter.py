
"""

IMAGE PROCESSING PROJECT CODE.

* PREPROCESS METADATA.

** CONCATENATE METADATA CSV FILES.

"""

import os
import sys

import pandas as pd
import numpy as np


def concatenate_csv(src_folder_path, dest_file_path="metadata.csv"):

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
