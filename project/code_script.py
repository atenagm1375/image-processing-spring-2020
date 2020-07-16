
"""

IMAGE PROCESSING PROJECT CODE.

* CODE SCRIPT

"""

# %% IMPORT MODULES

import gc
import os
from pathlib import Path

from data_preparation.csv_converter import benign_malignant_dataframe
from data_preparation.data_preparation import split_data, split_image_folders
from image_processing.image_processing import Data
from image_processing.models import albahar_model, train
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np

# %% ENVIRONMENT CONSTANTS

ISIC_PATH = "../../paper/data/ISICArchive/"
ORIGINAL_METADATA_PATH = "../../paper/data/metadata.csv"
DATA_PATH = "../DATA/"
MODEL_PATH = "../MODEL/"
SUB_METADATA = "metadata_"
SUB_BM = "bm_"
PART = "part_"
n_parts = 3
chunk_size = 128

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

    n_chunks = int(np.ceil(x_train.shape[0] / chunk_size))
    val_chunk_size = int(np.ceil(x_test.shape[0] // n_chunks))

    os.mkdir(DATA_PATH + PART + str(i) + "/x_train/")
    os.mkdir(DATA_PATH + PART + str(i) + "/y_train/")
    os.mkdir(DATA_PATH + PART + str(i) + "/x_test/")
    os.mkdir(DATA_PATH + PART + str(i) + "/y_test/")

    for j in range(n_chunks):
        np.save(DATA_PATH + PART + str(i) + f"/x_train/{j}.data",
                x_train[j * chunk_size:(j + 1) * chunk_size])
        np.save(DATA_PATH + PART + str(i) + f"/y_train/{j}.data",
                y_train[j * chunk_size:(j + 1) * chunk_size])
        np.save(DATA_PATH + PART + str(i) + f"/x_test/{j}.data",
                x_test[j * val_chunk_size:(j + 1) * val_chunk_size])
        np.save(DATA_PATH + PART + str(i) + f"/y_test/{j}.data",
                y_test[j * val_chunk_size:(j + 1) * val_chunk_size])

    del x_train, x_test, y_train, y_test, n_chunks, val_chunk_size
    gc.collect()

# %% PAPER CODE SCRIPT

print("=" * 100)

print("INITIALIZING MODEL...")
model = albahar_model(0.02)

print("SAVING MODEL...")
model_name = "albahar_1.h5"
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)
model.save(MODEL_PATH + model_name)

del model
gc.collect()

print("START TRAIN PROCESS...")
for i in range(1, n_parts + 1):
    print("-*" * 40)
    x_path_list = Path(DATA_PATH + PART + str(i) + "/x_train/").rglob("*.npy")
    y_path_list = Path(DATA_PATH + PART + str(i) + "/y_train/").rglob("*.npy")
    j = 1
    for x, y in zip(x_path_list, y_path_list):
        x_train = np.load(DATA_PATH + PART + str(i) + "/x_train.data.npy")
        y_train = np.load(DATA_PATH + PART + str(i) + "/y_train.data.npy")

        model = load_model(MODEL_PATH + model_name)

        history = train(model, x_train, y_train, optimizer=Adam(0.002))

        model.save(MODEL_PATH + model_name)

        np.save(f"history_{j}", history)
        j += 1
        print(history)

        del model, history
        gc.collect()

# %%
