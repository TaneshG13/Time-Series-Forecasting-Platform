import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)


class LSTMModel:

    def __init__(
        self,
        freq="Weekly",
        use_log=False
    ):

        self.freq = freq
        self.use_log = use_log

        self.lookback = (
            8 if freq == "Weekly"
            else 4
        )

        self.epochs = 100
        self.batch_size = 16

        SEED = 42

        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

    def build(
        self,
        n_features
    ):

        model = Sequential()

        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=(
                    self.lookback,
                    n_features
                )
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            LSTM(32)
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(1)
        )

        model.compile(
            optimizer='adam',
            loss='mse'
        )

        return model