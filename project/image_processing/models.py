
"""

IMAGE PROCESSING PROJECT CODE.

* MODELS.

"""

from keras import backend

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.regularizers import Regularizer
from keras.losses import categorical_crossentropy
from keras.metrics import AUC


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
        sigma = backend.sqrt((1 / self.n) *
                             ((backend.sum(x ** 2, axis=0) -
                               (1 / self.n) * (backend.sum(x, axis=0) ** 2))))
        return self.Lambda * backend.sum(sigma)


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
    conv1 = Conv2D(32, 3, strides=(3, 3), activation="relu",
                   kernel_regularizer=SDRegularizer(Lambda, 9))(inp)
    conv2 = Conv2D(64, 3, activation="relu", padding='same',
                   kernel_regularizer=SDRegularizer(Lambda, 9))(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout = Dropout(dropout_rate)(pool)
    flat = Flatten()(dropout)
    dense = Dense(128, activation="relu")(flat)
    out = Dense(2, activation="softmax")(dense)

    model = Model(inputs=inp, outputs=out)
    print(model.summary())

    return model


def pretrained_VGG16_model(n_classes=2, input_shape=(300, 300, 3),
                           weights='imagenet', trainable_layers=[]):
    """
    Model using VGG16 pretrained model.

    Parameters
    ----------
    n_classes : int, optional
        Number of output classes. The default is 2.
    input_shape : tuple, optional
        Shape of input images. The default is (300, 300, 3).
    weights : str, optional
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or
        the path to the weights file to be loaded.. The default is 'imagenet'.
    vgg_trainable_layers : list, optional
        List of layer indices to be trainable. The default is [].

    Returns
    -------
    model : keras.models.Sequential
        The model to be compiled and trained.

    """
    vgg = VGG16(weights=weights, include_top=False,
                input_shape=input_shape, classes=n_classes)

    model = Sequential(vgg.layers[:])
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    for ind, layer in enumerate(model.layers[:-3]):
        if ind not in trainable_layers:
            layer.trainable = False

    print(model.summary())

    return model


def train(model, x_train, y_train, x_val=None, y_val=None, epochs=100,
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
    callbacks : list of keras.callbacks, optional
        Callbacks used when fitting model to data. The default is None.

    Returns
    -------
    history : dict
        The history of loss and metrics in each epoch.

    """
    model.compile(loss=categorical_crossentropy, optimizer=optimizer,
                  metrics=["accuracy", "categorical_accuracy", AUC()])
    if x_val is None or y_val is None:
        history = model.fit(x_train, y_train, epochs=epochs,
                            callbacks=callbacks)
    else:
        history = model.fit(x_train, y_train, epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(x_val, y_val))
    return history.history
