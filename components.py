from model import *
from Extended_Min_Max import *
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

data = pd.read_csv('exchange.csv')

data['Date'] = pd.DatetimeIndex(data['Date'])
data.index = data['Date']
data = data.drop('Date', axis=1)


variables = list(data)

for var in variables:

    if var=='GBP_USD' or var == 'USD_CAD':
        method='gru'
    else:
        method = 'rnn'

    if var == 'USD_JPY':
        data[var] = np.log10(data[var])

    min_sample = len(data[var]) - 1460
    training = data[var].iloc[0:min_sample]
    test = data[var].iloc[min_sample:]

    normalization = Extended_Min_Max(training=training, test=test, days=len(test))
    training, test_norm = normalization.Normalization(features=[0, 1])

    data_preprocess= data_preparing(training, training, test_norm, test_norm, time_steps=5)
    trainX, valX, trainY, valY, testX, testY = data_preprocess.data_preprocessing()

    final_model = load_model('model/%s/model_%s_%s' %(method, var, 4)+'.h5')
    print(final_model.summary())
    comp = Model(final_model.layers[0].input, final_model.layers[10].output)
    conc = Model(final_model.layers[0].input, final_model.layers[9].output)

    conc_input = conc.predict(testX)

    all_importance = pd.DataFrame(data=conc_input, columns=['Swish',
                                'Linear', 'Sine', 'Cosine', 'Tanh', 'ReLU', 'Sigmoid'])

    all_importance['date'] = test.index.values[-len(all_importance['Swish']):]
    if var=='USD_JPY':
        all_importance[var] = 10** np.asarray(test[-len(all_importance['Swish']):])
    else:
        all_importance[var] = np.asarray(test[-len(all_importance['Swish']):])


    average_importance = all_importance[['Swish','Linear', 'Sine', 'Cosine', 'Tanh', 'ReLU', 'Sigmoid']].mean()
    print(average_importance)

    # all_importance.index = all_importance['date']
    # all_importance = all_importance.drop('date', axis=1)
    #
    # all_importance[['Swish', 'Linear', 'Sine', 'Cosine', 'Tanh', 'ReLU', 'Sigmoid']].plot(kind='area')
    # plt.show()

    all_importance.to_csv('importance/importance_%s.csv' %var, index=False)













