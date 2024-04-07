from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from consts import LSTM_N_UNITS, DENSE_N_UNITS, COLUMN_NAMES, DATA_PATH, \
    TARGET_VARIABLE, FEATURE_NAMES, CATEGORICAL_FEATURES, CHECKPOINT_FOLDER, COLUMNS_TO_DISPLAY_IN_PREDICTION_WEB, SEED
import os
import random

from lstm_model_training import get_model


def scale_data(X, Y) -> Tuple:
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X.values.astype('float32'))
    Y_scaled = scaler.fit_transform(Y.values)

    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    Y_scaled = Y_scaled.reshape((Y_scaled.shape[0],))

    return X_scaled, Y_scaled, scaler


def train_model(model,
                x_train,
                y_train,
                x_test,
                y_test,
                n_epochs: int,
                batch_size: int,
                model_checkpoint_path: str):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    training_hisotry = model.fit(x_train,
                                 y_train,
                                 epochs=n_epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_test, y_test),
                                 verbose=1,
                                 shuffle=False,
                                 callbacks=[cp_callback])

    pyplot.plot(training_hisotry.history['loss'], label='Training Loss')
    pyplot.plot(training_hisotry.history['val_loss'], label='Test Loss')
    pyplot.legend()
    pyplot.show()


def get_X_and_y(data: pd.DataFrame):
    X = data.loc[:, FEATURE_NAMES]
    y = data[TARGET_VARIABLE]
    return X, y


def load_model(model_checkpoint_path, dense_n_units, lstm_n_units, input_shape):
    model = get_model(dense_n_units=dense_n_units, lstm_n_units=lstm_n_units, input_shape=input_shape)
    model.load_weights(model_checkpoint_path)
    return model


def infer_model(model, x):
    predictions = model.predict(x)
    y_pred = tf.clip_by_value(tf.reshape(predictions, -1), clip_value_min=0, clip_value_max=np.inf)
    return pd.DataFrame(y_pred, columns=["week_points_pred"])


def random_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def infer(week_number: int) -> pd.DataFrame:
    model_checkpoint_path = str(Path(CHECKPOINT_FOLDER, str(week_number), f"season_2019_week_{week_number}"))
    data_df = pd.read_csv(DATA_PATH)

    test_data = data_df.query("season == 2019 and week_no == @week_number").reset_index()
    test_data_for_web = test_data[COLUMNS_TO_DISPLAY_IN_PREDICTION_WEB]
    test_data_for_model = test_data[COLUMN_NAMES]

    test_data_preprocessed = pd.get_dummies(test_data_for_model, columns=CATEGORICAL_FEATURES, drop_first=True)
    filled_in_columns = pd.DataFrame(0, index=np.arange(len(test_data_for_model)),
                                     columns=[f"week_no_{i}" for i in range(3, 39) if
                                              f"week_no_{i}" not in list(test_data_preprocessed.columns)])
    filled_in_columns[f"week_no_{week_number}"] = 1
    test_data_preprocessed = pd.concat([test_data_preprocessed, filled_in_columns], axis=1)
    x_test, y_test = get_X_and_y(data=test_data_preprocessed)

    x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)
    x_test_scaled, y_test_scaled, scaler = scale_data(x_test, y_test)

    input_shape = (1, x_test.shape[1])

    model = load_model(model_checkpoint_path=model_checkpoint_path,
                       dense_n_units=DENSE_N_UNITS,
                       lstm_n_units=LSTM_N_UNITS,
                       input_shape=input_shape)
    y_test_pred = infer_model(model=model, x=x_test_scaled)
    df = pd.concat([test_data_for_web, y_test_pred], axis=1)
    df = df.sort_values("week_points_pred", ascending=False)
    return df
