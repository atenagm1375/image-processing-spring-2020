
"""

IMAGE PROCESSING PROJECT CODE.

* MODELS.

"""

import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
from keras.regularizers import Regularizer
from keras.losses import binary_crossentropy


class SDRegularizer(Regularizer):
    """
    The Novel Regularizer offerred by Albahar 2019 paper.

    It penalizes dipersion of weights.
    """

    def __init__(self, Lambda, n):
        self.Lambda = Lambda
        self.n = n

    def __call__(self, x):
        """
        Compute a regularization penalty from an input tensor.

        Parameters
        ----------
        x : numpy.ndarray
            The weight matrix.

        Returns
        -------
        float
            The regularization amount.

        """
        sigma = np.sqrt((1 / self.n) *
                        ((np.sum(x ** 2, axis=0) -
                          (1 / self.n) * (np.sum(x, axis=0) ** 2))))
        return self.Lambda * np.sum(sigma)


def albahar_model(Lambda, input_shape=(300, 300, 3), dropout_rate=0.1):
    """
    Model structure proposed by Albahar.

    Parameters
    ----------
    Lambda : float
        Regularizer parameter.
    input_shape : tuple of int, optional
        Shape of input images. The default is (300, 300, 3).
    dropout_rate : float, optional
        Dropout layer rate. The default is 0.1.

    Returns
    -------
    model : keras.models.Model
        The model to be compiled and trained.

    """
    inp = Input(input_shape)
    conv1 = Conv2D(32, 3, activation="relu",
                   kernel_regularizer=SDRegularizer(Lambda, 9))(inp)
    conv2 = Conv2D(64, 3, activation="relu",
                   kernel_regularizer=SDRegularizer(Lambda, 9))(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout = Dropout(dropout_rate)(pool)
    flat = Flatten()(dropout)
    dense = Dense(128, activation="relu")(flat)
    out = Dense(2, activation="softmax")(dense)

    model = Model(inputs=inp, outputs=out)
    print(model.summary())

    return model


def train(model, x_train, x_val, y_train, y_val, epochs=100,
          optimizer=None, callbacks=None):
    """
    Compile and fit the model to data.

    Parameters
    ----------
    model : keras.models
        The model to train.
    x_train : numpy.ndarray
        Train data.
    x_val : numpy.ndarray
        Validation data.
    y_train : numpy.ndarray
        Train label.
    y_val : numpy.ndarray
        Validation label.
    epochs : int, optional
        Number of training epochs. The default is 100.
    optimizer : keras.optimizer, optional
        The optimizer to use for model compilation. The default is None.
    callbacks : keras.callbacks, optional
        Callbacks used when fitting model to data. The default is None.

    Returns
    -------
    history : dict
        The history of loss and metrics in each epoch.

    """
    model.compile(loss=binary_crossentropy, optimizer=optimizer,
                  metrics=["auc", "accuracy"])
    history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                        validation_data=(x_val, y_val))
    return history.history
