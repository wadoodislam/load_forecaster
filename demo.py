# from datetime import datetime as dt, timedelta
# import loadForecast as lf
import pandas as pd
import sm_forcaster as sm
import tensorflow as tf

from LoadForecaster import Forecaster

datafile = 'data/NCENT.csv'
useful_data = 'data/useful_data.csv'
df = pd.read_csv(datafile)

# all_x = sm.make_useful_df(df)
all_x = pd.read_csv(useful_data)
all_y = df['load']

train_x, train_y = all_x[:-8760], all_y[:-8760]
test_x, test_y = all_x[-8760:], all_y[-8760:]

# forecaster = sm.neural_net_model(train_x, train_y)
# train_predictions = [float(f) for f in forecaster.predict(train_x)]
# test_predictions = [float(f) for f in forecaster.predict(test_x)]
#
# train_accuracy = sm.MAPE(train_predictions, train_y)
# test_accuracy = sm.MAPE(test_predictions, test_y)
#
# print('Percent accuracy (MAPE). Train: {}.  Test: {}'.format(100-train_accuracy, 100-test_accuracy))
#
# forecaster.save('checkpoint/forecaster.h5')
# del forecaster
forecaster_reloaded = Forecaster('checkpoint/forecaster.h5')

train_predictions = [f for f in forecaster_reloaded.predict(train_x)]
test_predictions = [f for f in forecaster_reloaded.predict(test_x)]

train_accuracy = sm.MAPE(train_predictions, train_y)
test_accuracy = sm.MAPE(test_predictions, test_y)

print('Percent accuracy (MAPE). Train: {}.  Test: {}'.format(100-train_accuracy, 100-test_accuracy))
