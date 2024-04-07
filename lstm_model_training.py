from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional
import tensorflow as tf
from consts import N_EPOCHS, BATCH_SIZE, LSTM_N_UNITS, DENSE_N_UNITS, COLUMN_NAMES, DATA_PATH, \
    TARGET_VARIABLE, FEATURE_NAMES, CATEGORICAL_FEATURES, CHECKPOINT_FOLDER, WEEK_GAME_NUMBERS_2019, LR

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
                model_checkpoint_path: str
                ):
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


def get_model(dense_n_units, lstm_n_units, input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=lstm_n_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=lstm_n_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=lstm_n_units, input_shape=input_shape)))
    model.add(Dense(units=dense_n_units))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model


def get_X_and_y(data: pd.DataFrame):
    X = data.loc[:, FEATURE_NAMES]
    y = data[TARGET_VARIABLE]
    return X, y


def load_model(model_checkpoint_path, dense_n_units, lstm_n_units, input_shape):
    model = get_model(dense_n_units=dense_n_units, lstm_n_units=lstm_n_units, input_shape=input_shape)
    model.load_weights(model_checkpoint_path)
    return model


def infer_model(model, x):
    predictions = model(x)
    return tf.reshape(predictions, -1)

if __name__ == '__main__':
    for week_number in WEEK_GAME_NUMBERS_2019:
        hps = f"week_number: {week_number}, lr: {LR}, LSTM: {LSTM_N_UNITS}, BS:{BATCH_SIZE}"
        model_checkpoint_path = str(Path(CHECKPOINT_FOLDER, str(week_number), f"season_2019_week_{week_number}"))
        data_df = pd.read_csv(DATA_PATH)
        train_data = data_df.query("season == 2017 or season == 2018 or (season == 2019 and week_no < @week_number)")
        test_data = data_df.query("season == 2019 and week_no >= @week_number")
        test_data.reset_index(inplace=True)
        train_data = train_data[COLUMN_NAMES]
        test_data = test_data[COLUMN_NAMES]

        train_data_preprocessed = pd.get_dummies(train_data, columns=CATEGORICAL_FEATURES, drop_first=True)
        test_data_preprocessed = pd.get_dummies(test_data, columns=CATEGORICAL_FEATURES, drop_first=True)
        week_test_indication_df = pd.DataFrame(0, index=np.arange(len(test_data)),
                                               columns=[f"week_no_{i}" for i in range(3, 39) if
                                                        i not in range(week_number + 1, 9)])
        test_data_preprocessed = pd.concat([test_data_preprocessed, week_test_indication_df], axis=1)
        x_train, y_train = get_X_and_y(data=train_data_preprocessed)
        x_test, y_test = get_X_and_y(data=test_data_preprocessed)

        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_test, y_test = pd.DataFrame(x_test), pd.DataFrame(y_test)
        x_train_scaled, y_train_scaled, scaler = scale_data(x_train, y_train)
        x_test_scaled, y_test_scaled, scaler = scale_data(x_test, y_test)

        input_shape = (1, x_train.shape[1])

        model = get_model(dense_n_units=DENSE_N_UNITS, lstm_n_units=LSTM_N_UNITS, input_shape=input_shape)

        train_model(model=model,
                    x_train=x_train_scaled,
                    y_train=y_train,
                    x_test=x_test_scaled,
                    y_test=y_test,
                    n_epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    model_checkpoint_path=model_checkpoint_path)
