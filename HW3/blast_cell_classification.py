
"""

IMAGE PROCESSING HW3.

ASHENA G.MOHAMMADI-610398085

"""

# %% IMPORT MODULES

import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import pandas as pd
import PIL as pil
# from PIL import ImageOps
# import scipy.ndimage as sim
# from skimage import filters
import keras
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
import xgboost as xgb

# %% CONSTANTS

IMAGES_PATH = "../../HW3/ALL_IDB2/img/"

# %% LOAD DATA INTO A DATAFRAME

df = pd.DataFrame(columns=["id", "image", "x", "y"])

for file in os.listdir(IMAGES_PATH):
    name = file[:-6]
    label = file[-5]
    image = pil.Image.open(IMAGES_PATH + file)
    df = df.append({"id": name,
                    "image": image,
                    "x": np.array(image),
                    "y": int(label)}, ignore_index=True)

df = df.set_index("id")

# %% FUNCTIONS


def plot_loss_history(history):
    """
    Plot loss and accuracy of model during train.

    Parameters
    ----------
    history : dict
        The history of training the model.

    Returns
    -------
    None.

    """
    loss = history.get('loss')
    val_loss = history.get('val_loss')

    acc = history.get('accuracy')
    val_acc = history.get('val_accuracy')

    _, axes = plt.subplots(2)
    axes[0].plot(list(range(len(loss))), loss, label="train loss")
    axes[0].plot(list(range(len(val_loss))), val_loss, label="validation loss")
    axes[0].legend()

    axes[1].plot(list(range(len(acc))), acc, label="train accuracy")
    axes[1].plot(list(range(len(val_acc))), val_acc, label="test accuracy")
    axes[1].legend()

    plt.show()


def get_scores(y_true, y_pred):
    """
    Compute confusion matrix and print classification report.

    Parameters
    ----------
    y_true : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns
    -------
    numpy.ndarray
        mean confusion matrix.

    """
    print("#" * 40)
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred))
    print("-" * 20)
    print("CONFUSION MATRIX:")
    conf = confusion_matrix(y_true, y_pred)
    print(conf)
    print("#" * 40)
    return conf


def use_vgg16(x, y, input_shape, plot_loss=True):
    """
    Use VGG16 network to classify the blast cells.

    Parameters
    ----------
    x : numpy.ndarray
        The images.
    y : numpy.ndarray
        The labels.
    input_shape : tuple
        Shape of network input.
    plot_loss : bool, optional
        Whether to plot the loss during training. The default is True.

    Returns
    -------
    numpy.ndarray
        mean confusion matrix.

    """
    x = np.stack(x)
    y = y.astype(int)
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=0.2,
                                                      random_state=97)
    vgg = keras.applications.vgg16.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_shape=input_shape)
    model = keras.models.Sequential(vgg.layers[:8])
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    for layer in model.layers[:8]:
        layer.trainable = False

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=8,
                        callbacks=[
                            keras.callbacks.ReduceLROnPlateau(patience=5)],
                        validation_data=(x_val, y_val))

    if plot_loss:
        plot_loss_history(history.history)

    y_pred = np.ravel(np.round(model.predict(np.stack(x_val)))).astype(int)
    return get_scores(y_val, y_pred)


def _vgg16_classifier_model(x_train, x_val, y_train, y_val,
                            input_shape, classifier, trainable=False):
    vgg = keras.applications.vgg16.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_shape=input_shape)

    if trainable:
        model = keras.models.Sequential(vgg.layers[:-1])
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        for layer in model.layers[:-2]:
            layer.trainable = False

        print(model.summary())

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.RMSprop(learning_rate=0.01),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=10,
                            callbacks=[
                                keras.callbacks.EarlyStopping(patience=1)],
                            validation_data=(x_val, y_val))

        plot_loss_history(history.history)

        feature_extractor = keras.models.Sequential(model.layers[:-1])
    else:
        feature_extractor = keras.models.Sequential(vgg.layers[:-1])
        feature_extractor.add(keras.layers.Flatten())

    train_feats = feature_extractor.predict(x_train)
    val_feats = feature_extractor.predict(x_val)

    classifier.fit(train_feats, y_train)
    return classifier.predict(val_feats)


def use_vgg16_with_classifier(x, y, input_shape, classifier,
                              cv=True, trainable=False):
    """
    Use VGG16 network to extract feature and feed into a classifier.

    Parameters
    ----------
    x : numpy.ndarray
        The images.
    y : numpy.ndarray
        The labels.
    input_shape : tuple
        Shape of network input.
    classifier : sklearn.<classifier_parent>._classes.<classifier>
        The classifier to be used.
    cv : bool, optional
        Whether to use cross validation or not. The default is True.
    trainable : bool, optional
        Whether to train some part of VGG network or not. The default is False.

    Returns
    -------
    numpy.ndarray
        mean confusion matrix.

    """
    x = np.stack(x)
    y = y.astype(int)
    if cv:
        scores = []
        folds = KFold(5, shuffle=True, random_state=97)
        for trn_idx, tst_idx in folds.split(x):
            x_train, y_train = x[trn_idx], y[trn_idx]
            x_val, y_val = x[tst_idx], y[tst_idx]
            y_pred = _vgg16_classifier_model(x_train, x_val,
                                             y_train, y_val,
                                             input_shape, classifier,
                                             trainable=trainable)
            scores.append(get_scores(y_val, y_pred))
        return np.mean(scores, axis=0)
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                          test_size=0.2,
                                                          random_state=97)
        y_pred = _vgg16_classifier_model(x_train, x_val,
                                         y_train, y_val,
                                         input_shape, classifier,
                                         trainable=trainable)
        return get_scores(y_val, y_pred)

# %% RUN MODEL INCLUDING VGG16 WITH SIMPLE HOLDOUT METHOD


df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
print("USING FIRST 8 LAYERS OF VGG16 AND ADDING DROPOUT(0.2) AND DNESE(128)"
      " FOLLOWING A SIGMOID UNIT:")
conf_mat = use_vgg16(df["x"].values, df["y"].values, image_shape)

print(conf_mat)
print("ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
ACCURACY: 0.8653846153846154

"""

# %% RUN MODEL WITH NON-TRAINABLE VGG16 AND SVM(RBF KERNEL) WITH CV=5

df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
print("USING VGG16 FEATURE EXTRACTION:")
conf_mat = use_vgg16_with_classifier(df["x"].values, df["y"].values,
                                     image_shape, classifier=SVC(kernel='rbf',
                                                                 C=5))
print(conf_mat)
print("MEAN ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
MEAN ACCURACY: 0.9038461538461539

"""

# %% RUN MODEL WITH TRAINABLE VGG16 AND SVM(RBF KERNEL) WITH CV=5

df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
print("USING VGG16 FEATURE EXTRACTION:")
conf_mat = use_vgg16_with_classifier(df["x"].values, df["y"].values,
                                     image_shape, classifier=SVC(kernel='rbf',
                                                                 C=1),
                                     trainable=True)
print(conf_mat)
print("MEAN ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
MEAN ACCURACY: 0.9230769230769231

"""

# %% RUN MODEL WITH TRAINABLE VGG16 AND XGBOOST WITH CV=5

df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
params = {
    'n_estimators': 200,
    'eta': 0.1,
    'max_depth': 5,
    'reg_lambda': 0.02,
    'objective': 'binary:logistic',
    'booster': 'dart',
    'gamma': 0.2,
    'colsample_bytree': 0.35,
    'subsample': 0.75
    }
print("USING VGG16 FEATURE EXTRACTION:")
conf_mat = use_vgg16_with_classifier(df["x"].values, df["y"].values,
                                     image_shape, trainable=True,
                                     classifier=xgb.XGBClassifier(**params))
print(conf_mat)
print("MEAN ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
MEAN ACCURACY: 0.9115384615384616

"""

# %% RUN MODEL WITH TRAINABLE VGG16 AND XGBOOST WITH CV=5

df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
params = {
    'n_estimators': 200,
    'eta': 0.1,
    'max_depth': 5,
    'reg_lambda': 0.05,
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'gamma': 0.1,
    'colsample_bytree': 0.35,
    'subsample': 0.75
    }
print("USING VGG16 FEATURE EXTRACTION:")
conf_mat = use_vgg16_with_classifier(df["x"].values, df["y"].values,
                                     image_shape, trainable=True,
                                     classifier=xgb.XGBClassifier(**params))
print(conf_mat)
print("MEAN ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
MEAN ACCURACY: 0.9153846153846152

"""

# %% RUN MODEL WITH NON-TRAINABLE VGG16 AND XGBOOST WITH CV=5

df["x"] = df["image"].apply(lambda img: np.array(
    img.resize((200, 200), pil.Image.ANTIALIAS)))
image_shape = df["x"].iloc[0].shape
params = {
    'n_estimators': 1000,
    'eta': 0.1,
    'max_depth': 5,
    'reg_lambda': 0.05,
    'objective': 'binary:logistic',
    'booster': 'dart',
    'gamma': 0.1,
    'colsample_bytree': 0.7,
    'subsample': 0.8
    }
print("USING VGG16 FEATURE EXTRACTION:")
conf_mat = use_vgg16_with_classifier(df["x"].values, df["y"].values,
                                     image_shape,
                                     classifier=xgb.XGBClassifier(**params))
print(conf_mat)
print("MEAN ACCURACY:", (conf_mat[0][0] + conf_mat[1][1])/np.sum(conf_mat))
"""

OUTPUT:
MEAN ACCURACY:

"""
# %%
