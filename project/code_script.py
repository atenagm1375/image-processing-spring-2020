
"""

IMAGE PROCESSING PROJECT CODE.

* CODE SCRIPT

"""

# %% IMPORT MODULES

import gc

from data_preparation.csv_converter import benign_malignant_dataframe
from data_preparation.data_preparation import split_data, split_image_folders
from image_processing.image_processing import Data
from image_processing.models import *
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import numpy as np

# %% ENVIRONMENT CONSTANTS

ISIC_PATH = "../../paper/data/ISICArchive/"
ORIGINAL_METADATA_PATH = "../../paper/data/metadata.csv"
DATA_PATH = "../DATA/"
SUB_METADATA = "metadata_"
SUB_BM = "bm_"
PART = "part_"
n_parts = 3

# %% SPLIT DATA

split_data(ORIGINAL_METADATA_PATH, n_parts=n_parts, save_to=DATA_PATH)

meta_paths = [DATA_PATH + SUB_METADATA + str(i) + ".csv"
              for i in range(1, n_parts + 1)]
split_image_folders(meta_paths, ISIC_PATH, DATA_PATH)

# %% GENERATE BENIGN/MALIGNANT DATAFRAME

for i in range(1, n_parts + 1):
    benign_malignant_dataframe(ORIGINAL_METADATA_PATH, encoder='one_hot',
                               indeterminant="remove_all",
                               save_to=DATA_PATH + SUB_BM + str(i) + ".csv")

# %% SPLIT DATA

for i in range(1, n_parts + 1):
    print("#" * 80)
    print(f"Trial {i}:")
    data = Data()
    data.set_x(DATA_PATH + PART + str(i) + "/", (300, 300))
    data.set_y(DATA_PATH + SUB_BM + str(i) + ".csv")

    x_train, x_test, y_train, y_test = train_test_split(data.x, data.y,
                                                        test_size=0.3)

    del data.x, data.y, data.names, data
    gc.collect()

    np.save(DATA_PATH + PART + str(i) + f"/x_train.data", x_train)
    np.save(DATA_PATH + PART + str(i) + f"/y_train.data", y_train)
    np.save(DATA_PATH + PART + str(i) + f"/x_test.data", x_test)
    np.save(DATA_PATH + PART + str(i) + f"/y_test.data", y_test)

    del x_train, x_test, y_train, y_test
    gc.collect()

# %% PAPER CODE SCRIPT

print("=" * 100)
print("START TRAIN PROCESS...")

for i in range(1, n_parts + 1):
    print("-*" * 40)
    x_train = np.load(DATA_PATH + PART + str(i) + "/x_train.data.npy")
    y_train = np.load(DATA_PATH + PART + str(i) + "/y_train.data.npy")

    model = albahar_model(0.02)

    history = train(model, x_train, y_train, optimizer=Adam(0.002))

    print(history)

    del model, history
    gc.collect()

# %% USE PRETRAINED VGG

print("=" * 100)
print("START TRAIN PROCESS...")

for i in range(1, n_parts + 1):
    print("-*" * 40)
    x_train = np.load(DATA_PATH + PART + str(i) + "/x_train.data.npy")
    y_train = np.load(DATA_PATH + PART + str(i) + "/y_train.data.npy")
    x_test = np.load(DATA_PATH + PART + str(i) + "/x_test.data.npy")
    y_test = np.load(DATA_PATH + PART + str(i) + "/y_test.data.npy")

    model = pretrained_VGG16_model()

    history = train(model, x_train, y_train, x_test, y_test,
                    optimizer=Adam(0.0005), epochs=8)

    print(history)

    del model, history
    gc.collect()

# %%
