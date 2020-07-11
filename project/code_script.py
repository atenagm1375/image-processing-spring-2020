
"""

IMAGE PROCESSING PROJECT CODE.

* CODE SCRIPT

"""

# %% IMPORT MODULES

import gc

from image_processing.image_processing import Data
from image_processing.models import albahar_model, train
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# %% ENVIRONMENT CONSTANTS

DATA_PATH = "../DATA/"
METADATA = "bm_"
PART = "part_"
n_parts = 3

# %% PAPER CODE SCRIPT

for i in range(1, n_parts):
    print("#" * 40)
    print(f"Trial {i}:")
    data = Data()
    data.set_x(DATA_PATH + PART + str(i) + "/", (300, 300))
    data.set_y(DATA_PATH + METADATA + str(i) + ".csv")

    # data.powerlaw_transform(1.5, 256)

    model = albahar_model(0.02)

    x_train, x_test, y_train, y_test = train_test_split(data.x, data.y,
                                                        test_size=0.3)

    del data.x
    del data.y
    del data.names
    del data
    gc.collect()

    history = train(model, x_train, x_test, y_train, y_test,
                    optimizer=Adam(0.002))

    print(history)
    print("#" * 40)

# %%
