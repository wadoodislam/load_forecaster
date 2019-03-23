# from datetime import datetime as dt, timedelta
import pandas as pd
import loadForecast as lf
import sm_forcaster as sm

f = 'data/NCENT.csv'
df = pd.read_csv(f)
all_X = sm.make_useful_df(df)
all_y = df['load']
predictions, accuracy = sm.neural_net_predictions(all_X, all_y)
print('Percent accuracy (MAPE). Train: {}.  Test: {}'.format(100-accuracy['train'], 100-accuracy['test']))
#
# df_r = pd.DataFrame()
# df_r['predicted_load'] = predictions
# df_r['actual_load'] = [float(f) for f in all_y[-8760:]]
# df_r.index = [dt(2018, 1, 1, 0) + timedelta(hours=1)*i for i in range(8760)]
# df_r.plot(figsize=(10, 3), title="NCENT Texas 2018: prediction v. actual load")
