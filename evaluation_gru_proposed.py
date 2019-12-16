import pandas as pd
from model import *
from Extended_Min_Max import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
import math

def evaluation(true_Y, pred_Y):
    rmse = math.sqrt(mean_squared_error(true_Y, pred_Y))
    mae = (mean_absolute_error(true_Y, pred_Y))
    smape = np.mean(np.abs((true_Y - pred_Y) / (true_Y + pred_Y)))
    r_squared = r2_score(true_Y, pred_Y)

    return rmse, mae, smape, r_squared


data = pd.read_csv('exchange.csv')

data['Date'] = pd.DatetimeIndex(data['Date'])
data.index = data['Date']
data = data.drop('Date', axis=1)
method='gru'

variables = list(data)
for var in variables:
    print(var)

    if var == 'USD_JPY':
        data[var] = np.log10(data[var])

    min_sample = len(data[var]) - 1825
    min_train = data[var].iloc[0:min_sample]
    train = data[var].iloc[min_sample:]
    tscv = TimeSeriesSplit(n_splits=4)

    performance = {'Data': [], 'Method': [], 'Index': [],
                   'RMSE': [], 'MAE': [], 'sMAPE': [], 'R_squared': []}

    i = 1
    for train_index, test_index in tscv.split(train):

        training = pd.concat([min_train, train[train_index]], axis=0)
        test = train[test_index]

        #if max(training)<max(test) or min(training)>min(test):

        normalization = Extended_Min_Max(training=training, test=test, days=len(test))

        #training_relu, test_relu = normalization.sklearn_normalization(features=[0, 1])
        #training_tanh, test_tanh = normalization.sklearn_normalization(features=[-1, 1])
        training, test = normalization.Normalization(features=[0, 1])

        data_preprocess = data_preparing(training, training, test, test, time_steps=5)
        trainX, valX, trainY, valY, testX, testY = data_preprocess.data_preprocessing()

        modelling = Models(trainX,  valX, trainY, valY,
                 rnn_layers=[16, 1], time_steps=5, batch_size=64,
                 early_stopping=100, model_path='model/%s/model_%s_%s' %(method, var, i),
                method=method, var=var, epoch=3000, lr=0.001,
                           reg_lambda= 0.00001)

        final_model = modelling.proposed_model_gru()
        print(final_model.summary())
        pred_y = final_model.predict(testX)

        #testY = normalization.sklearn_DE_normalization(testY)
        #pred_y = normalization.sklearn_DE_normalization(pred_y)
        testY = normalization.DE_normalization(testY)
        pred_y = normalization.DE_normalization(pred_y)

        if var == 'USD_JPY':
            testY = 10 ** testY
            pred_y = 10 ** pred_y

        rmse, mae, smape, r_squared = evaluation(true_Y=testY, pred_Y=pred_y)

        performance['Data'].append(var)
        performance['Method'].append('Proposed_model %s' %method)
        performance['Index'].append(i)
        performance['RMSE'].append(rmse)
        performance['MAE'].append(mae)
        performance['sMAPE'].append(smape)
        performance['R_squared'].append(r_squared)

        #else:
         #   print ('does not need to train' )

        i = i + 1

    res = pd.DataFrame.from_dict(performance)
    res.to_csv('model/%s/performance_up.csv' %method, index=False, mode='a')








