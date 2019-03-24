"""
Neural net implementation of electric load forecasting.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import zscore


def add_noise(m, std):
    noise = np.random.normal(0, std, m.shape[0])
    return m + noise


def make_useful_df(data, noise=2.5, hours_prior=24):
    result_df = pd.DataFrame()
    result_df['dates'] = data.apply(lambda x: datetime(
                                        int(x['year']),
                                        int(x['month']),
                                        int(x['day']),
                                        int(x['hour'])),
                                    axis=1)
    # create day of week vector: 0 is Monday.
    day_labels = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    for index, day in enumerate(day_labels):
        result_df[day] = (result_df["dates"].dt.dayofweek == index).astype(int)

    # create hour of day vector
    hour_labels = [("h" + str(i)) for i in range(24)]
    for index, hour in enumerate(hour_labels):
        result_df[hour] = (result_df["dates"].dt.hour == index).astype(int)

    # create month vector
    month_labels = [("m" + str(i)) for i in range(1, 12 + 1)]
    for index, month in enumerate(month_labels):
        result_df[month] = (result_df["dates"].dt.month == index).astype(int)

    temperature = data["tempc"].replace([-9999], np.nan)
    temperature.ffill(inplace=True)

    # day-before predictions temperature
    temperature_predictions = add_noise(temperature, noise)
    result_df["temp_n"] = zscore(temperature_predictions)
    result_df['temp_n^2'] = result_df["temp_n"] ** 2

    result_df["load_n"] = zscore(data["load"])
    result_df["years_n"] = zscore(result_df["dates"].dt.year)

    # add the value of the load 24hrs before
    result_df["load_prev_n"] = result_df["load_n"].shift(hours_prior)
    result_df["load_prev_n"].bfill(inplace=True)

    return result_df.drop(["dates", "load_n"], axis=1)


def neural_net_model(train_x, train_y, EPOCHS=10):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(train_x.shape[1], activation=tf.nn.relu, input_shape=[len(train_x.keys())]),
        tf.keras.layers.Dense(train_x.shape[1], activation=tf.nn.relu),
        tf.keras.layers.Dense(train_x.shape[1], activation=tf.nn.relu),
        tf.keras.layers.Dense(train_x.shape[1], activation=tf.nn.relu),
        tf.keras.layers.Dense(train_x.shape[1], activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    history = model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop],
    )

    return model


def MAPE(predictions, answers):
    assert len(predictions) == len(answers)
    return sum([abs(x - y) / (y + 1e-5) for x, y in zip(predictions, answers)]) / len(answers) * 100

