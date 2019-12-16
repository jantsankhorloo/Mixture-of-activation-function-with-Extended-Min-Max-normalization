from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt


class Extended_Min_Max:

    def __init__(self, training, test, days, alpha=0.025):

        self.training = training
        self.test = test
        self.days = days
        self.alpha = alpha


    def VaR(self):
    
        y1=self.training.pct_change()
        y1=y1.dropna()

        var = np.sqrt(0.94 * np.var(y1) + 0.06 * (y1[-1] ** 2))
        
        return var
        
    def Global_Min_Max(self):

        MAX_VAR=-stats.norm.ppf(self.alpha)*self.VaR()*np.sqrt(self.days)+self.training[-1]
        MAX_his = np.max(self.training)

        MAX = max(MAX_VAR, MAX_his)

        MIN_VAR = stats.norm.ppf(self.alpha)*self.VaR()*np.sqrt(self.days)+self.training[-1]
        MIN_his = min(self.training)


        MIN = min(MIN_VAR, MIN_his)

        return  MAX, MIN
        

    def Normalization(self, features=None):

        self.MAX, self.MIN = self.Global_Min_Max()

        global normalized_train, normalized_test

        if features is None:
            features = [0, 1]

        if features[0]==0:
            normalized_train = (self.training-self.MIN)/(self.MAX-self.MIN)
            normalized_test = (self.test - self.MIN) / (self.MAX - self.MIN)
        elif features[0]==-1:
            normalized_train = 2*((self.training - self.MIN) / (self.MAX - self.MIN))-1
            normalized_test = 2*((self.test - self.MIN) / (self.MAX - self.MIN))-1

        return normalized_train, normalized_test

    def sklearn_normalization(self, features=None):

        global normalized_train, normalized_test

        if features is None:
            features = [0, 1]

        if features[0] == 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scalar = scaler.fit(np.expand_dims(self.training, axis=1))
            normalized_train = train_scalar.transform(np.expand_dims(self.training, axis=1))
            normalized_test = train_scalar.transform(np.expand_dims(self.test, axis=1))
        elif features[0] == -1:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train_scalar = scaler.fit(np.expand_dims(self.training, axis=1))
            normalized_train = train_scalar.transform(np.expand_dims(self.training, axis=1))
            normalized_test = train_scalar.transform(np.expand_dims(self.test, axis=1))

        return normalized_train, normalized_test


    def DE_normalization(self, norm_data, features=None):

        global data

        if features is None:
            features = [0, 1]

        if features[0] == 0:
            data = (self.MAX-self.MIN)*norm_data+self.MIN
        elif features[0] == -1:
            data = (self.MAX - self.MIN) * (norm_data+1) + self.MIN

        return data

    def sklearn_DE_normalization(self, norm_data, features=None):

        global data

        if features is None:
            features = [0, 1]

        if features[0] == 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scalar = scaler.fit(np.expand_dims(self.training, axis=1))
            data = train_scalar.inverse_transform(norm_data)

        elif features[0] == -1:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            train_scalar = scaler.fit(np.expand_dims(self.training, axis=1))
            data = train_scalar.inverse_transform(norm_data)

        return data

'''
data = pd.read_csv('exchange.csv')

data['Date'] = pd.DatetimeIndex(data['Date'])
data.index = data['Date']
data = data.drop('Date', axis=1)
method = 'rnn'

variables = list(data)
for var in variables:
    print(var)
    if np.min(data[var])>10:
        data[var] = np.log10(data[var])

    min_sample = len(data[var]) - 5000
    min_train = data[var].iloc[0:min_sample]
    train = data[var].iloc[min_sample:]

    tscv = TimeSeriesSplit(n_splits=4999)

    i = 7
    k = 30
    l = 365
    test_value = []
    max_value_week = []
    min_value_week = []
    max_value_month = []
    min_value_month = []
    max_value_year = []
    min_value_year = []

    ex_week=0
    ex_month = 0
    ex_year = 0

    for train_index, test_index in tscv.split(train):

        train_set = min_train
        test_set = train[test_index]
        print(min_train[-l:])

        normalization = Extended_Min_Max(training=train_set, test=test_set, days=i)
        max1, min1 = normalization.Global_Min_Max()
        normalization = Extended_Min_Max(training=train_set, test=test_set, days=k)
        max2, min2 = normalization.Global_Min_Max()
        normalization = Extended_Min_Max(training=train_set, test=test_set, days=l)
        max3, min3 = normalization.Global_Min_Max()

        test_value.append(test_set)
        max_value_week.append(max1)
        min_value_week.append(min1)
        max_value_month.append(max2)
        min_value_month.append(min2)
        max_value_year.append(max3)
        min_value_year.append(min3)

        if max1<max(min_train[-i:]) or min1>min(min_train[-i:]):
            ex_week = ex_week +1

        if max2<max(min_train[-k:]) or min2>min(min_train[-k:]):
            ex_month = ex_month +1

        if max3 < max(min_train[-l:]) or min3 > min(min_train[-l:]):
            ex_year = ex_year + 1


        #i=i+1
        min_train = pd.concat([min_train, test_set])
    date = min_train[-len(test_value):].index.values
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title('Value at Risk of %s' %var)
    ax.xaxis.grid()
    ax.plot(date, test_value, 'black', label='Actual value')
    ax.plot(date, max_value_week, 'red', label='Maximum weekly bound')
    ax.plot(date, min_value_week, 'red', label='Minimum weekly bound')
    ax.plot(date, max_value_month, 'blue', label='Maximum monthly bound')
    ax.plot(date, min_value_month, 'blue', label='Minimum monthly bound')
    ax.plot(date, max_value_year, 'grey', label='Maximum yearly bound')
    ax.plot(date, min_value_year, 'grey', label='Minimum yearly bound')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()

    print(ex_week, ex_month, ex_year, len(test_value))

'''


