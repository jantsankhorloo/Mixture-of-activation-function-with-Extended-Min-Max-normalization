from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import adam
from keras.models import load_model
from keras import backend as K
from keras.models import Model
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects
import os
import numpy as np


import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

np.random.seed(6)

class Sine(Activation):

    def __init__(self, activation, **kwargs):
        super(Sine, self).__init__(activation, **kwargs)
        self.__name__ = 'sine'

def sine(x):

    return K.sin(x)

get_custom_objects().update({'sine': Sine(sine)})


class Cos(Activation):

    def __init__(self, activation, **kwargs):
        super(Cos, self).__init__(activation, **kwargs)
        self.__name__ = 'cos'


def cos(x):
    return K.cos(x)


get_custom_objects().update({'cos': Cos(cos)})


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x):
    return K.sigmoid(x)*x

get_custom_objects().update({'swish': Swish(swish)})


class LSTM_model():

    def __init__(self, inputs, layers, time_steps,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.time_steps = time_steps
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def LSTM_model(self, act_name):

        input_shape = (self.time_steps, int(self.inputs.shape[2]))

        LSTM_model = Sequential(name=act_name)

        if len(self.layers) > 1:
            LSTM_model.add(LSTM(self.layers[0], activation=self.activation, return_sequences=True,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    LSTM_model.add(LSTM(self.layers[layer], activation=self.activation, return_sequences=True,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
                else:
                    LSTM_model.add(LSTM(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            LSTM_model.add(LSTM(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        lstm_output = LSTM_model(self.inputs)

        return lstm_output

class GRU_model():

    def __init__(self, inputs, layers, time_steps,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.time_steps = time_steps
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def GRU_model(self, act_name):

        input_shape = (self.time_steps, int(self.inputs.shape[2]))

        GRU_model = Sequential(name=act_name)

        if len(self.layers) > 1:
            GRU_model.add(GRU(self.layers[0], activation=self.activation, return_sequences=True,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    GRU_model.add(GRU(self.layers[layer], activation=self.activation, return_sequences=True,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
                else:
                    GRU_model.add(GRU(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            GRU_model.add(GRU(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        gru_output = GRU_model(self.inputs)

        return gru_output

class RNN_model():

    def __init__(self, inputs, layers, time_steps,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.time_steps = time_steps
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def RNN_model(self, act_name):

        input_shape = (self.time_steps, int(self.inputs.shape[2]))

        RNN_model = Sequential(name=act_name)

        if len(self.layers) > 1:
            RNN_model.add(SimpleRNN(self.layers[0], activation=self.activation, return_sequences=True,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    RNN_model.add(SimpleRNN(self.layers[layer], activation=self.activation, return_sequences=True,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
                else:
                    RNN_model.add(SimpleRNN(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda),
                                      recurrent_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            RNN_model.add(SimpleRNN(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              recurrent_regularizer=regularizers.l1(self.reg_lambda),
                              input_shape=input_shape))

        rnn_output = RNN_model(self.inputs)

        return rnn_output


class MLP_model():

    def __init__(self, inputs, layers,
                 activation, reg_lambda= 0.00001):

        self.inputs = inputs
        self.layers = layers
        self.activation = activation
        self.reg_lambda = reg_lambda

    def model(self, act_name):

        input_dim = int(self.inputs.shape[1])

        MLP_MODEL = Sequential(name=act_name)

        if len(self.layers) > 1:
            MLP_MODEL.add(Dense(self.layers[0], activation=self.activation,
                              kernel_regularizer=regularizers.l1(self.reg_lambda),
                              input_dim=input_dim))
            for layer in range(1, len(self.layers)):
                if layer < len(self.layers) - 1:
                    MLP_MODEL.add(Dense(self.layers[layer], activation=self.activation,
                                      kernel_regularizer=regularizers.l1(self.reg_lambda)))
        else:
            MLP_MODEL.add(Dense(self.layers[0], activation=self.activation,
                                kernel_regularizer=regularizers.l1(self.reg_lambda),
                                input_dim=input_dim))

        #MLP_MODEL.add((Dense(1, activation=self.activation)))

        mlp_output = MLP_MODEL(self.inputs)

        return mlp_output


class Models():

    def __init__(self, trainingX,  valX, trainingY, valY,
                 rnn_layers, time_steps, batch_size,
                 early_stopping, model_path, method, var, epoch, lr, reg_lambda= 0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.rnn_layers = rnn_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[2]
        self.batch_size = batch_size
        self.epoch = epoch
        self.var = var
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.method = method

    def proposed_model_rnn(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s'
                            %(self.method, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s'
                                     %(self.method, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_rnn = Input(shape=(self.time_steps, self.array_size,))

            mlp_model_exp = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'swish')
            output_exp = mlp_model_exp.RNN_model('swish')
            model_exp = Model(inputs=input_rnn, outputs=output_exp,  name='swish_loss')
            output1 = model_exp(input_rnn)

            mlp_model_line = RNN_model(input_rnn, self.rnn_layers, self.time_steps,'linear')
            output_line = mlp_model_line.RNN_model('linear')
            model_line = Model(inputs=input_rnn, outputs=output_line, name='linear_loss')
            output2 = model_line(input_rnn)

            mlp_model_sine = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'sine')
            output_sine = mlp_model_sine.RNN_model('sine')
            model_sine = Model(inputs=input_rnn, outputs=output_sine, name='sine_loss')
            output3 = model_sine(input_rnn)

            mlp_model_cos = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'cos')
            output_cos = mlp_model_cos.RNN_model('cos')
            model_cos = Model(inputs=input_rnn, outputs=output_cos, name='cos_loss')
            output4 = model_cos(input_rnn)

            mlp_model_tanh = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'tanh')
            output_tanh = mlp_model_tanh.RNN_model('tanh')
            model_tanh = Model(inputs=input_rnn, outputs=output_tanh, name='tanh_loss')
            output5 = model_tanh(input_rnn)

            mlp_model_relu = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'relu')
            output_relu = mlp_model_relu.RNN_model('relu')
            model_relu = Model(inputs=input_rnn, outputs=output_relu, name='relu_loss')
            output6 = model_relu(input_rnn)

            mlp_model_sigmoid = RNN_model(input_rnn, self.rnn_layers, self.time_steps, 'sigmoid')
            output_sigmoid = mlp_model_sigmoid.RNN_model('sigmoid')
            model_sigmoid = Model(inputs=input_rnn, outputs=output_sigmoid, name='sigmoid_loss')
            output7 = model_sigmoid(input_rnn)

            final_input = Concatenate()([output1, output2, output3, output4,
                                         output5, output6, output7])

            soft_output = Dense(7, activation='softmax')(final_input)

            final_input = Multiply()([final_input, soft_output])

            final_output = Dense(1, activation='linear')(final_input)

            proposed_model = Model(inputs=input_rnn, outputs=final_output, name='Prediction_loss')

            print(proposed_model.summary())

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

    def proposed_model_lstm(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s'
                            %(self.method, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s'
                                     %(self.method, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_rnn = Input(shape=(self.time_steps, self.array_size,))

            mlp_model_exp = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'swish')
            output_exp = mlp_model_exp.LSTM_model('swish')
            model_exp = Model(inputs=input_rnn, outputs=output_exp,  name='swish_loss')
            output1 = model_exp(input_rnn)

            mlp_model_line = LSTM_model(input_rnn, self.rnn_layers, self.time_steps,'linear')
            output_line = mlp_model_line.LSTM_model('linear')
            model_line = Model(inputs=input_rnn, outputs=output_line, name='linear_loss')
            output2 = model_line(input_rnn)

            mlp_model_sine = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'sine')
            output_sine = mlp_model_sine.LSTM_model('sine')
            model_sine = Model(inputs=input_rnn, outputs=output_sine, name='sine_loss')
            output3 = model_sine(input_rnn)

            mlp_model_cos = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'cos')
            output_cos = mlp_model_cos.LSTM_model('cos')
            model_cos = Model(inputs=input_rnn, outputs=output_cos, name='cos_loss')
            output4 = model_cos(input_rnn)

            mlp_model_tanh = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'tanh')
            output_tanh = mlp_model_tanh.LSTM_model('tanh')
            model_tanh = Model(inputs=input_rnn, outputs=output_tanh, name='tanh_loss')
            output5 = model_tanh(input_rnn)

            mlp_model_relu = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'relu')
            output_relu = mlp_model_relu.LSTM_model('relu')
            model_relu = Model(inputs=input_rnn, outputs=output_relu, name='relu_loss')
            output6 = model_relu(input_rnn)

            mlp_model_sigmoid = LSTM_model(input_rnn, self.rnn_layers, self.time_steps, 'sigmoid')
            output_sigmoid = mlp_model_sigmoid.LSTM_model('sigmoid')
            model_sigmoid = Model(inputs=input_rnn, outputs=output_sigmoid, name='sigmoid_loss')
            output7 = model_sigmoid(input_rnn)

            final_input = Concatenate()([output1, output2, output3, output4,
                                         output5, output6, output7])

            soft_output = Dense(7, activation='softmax')(final_input)

            final_input = Multiply()([final_input, soft_output])

            final_output = Dense(1, activation='linear')(final_input)

            proposed_model = Model(inputs=input_rnn, outputs=final_output, name='Prediction_loss')

            print(proposed_model.summary())

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

    def proposed_model_gru(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s'
                            %(self.method, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s'
                                     %(self.method, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_rnn = Input(shape=(self.time_steps, self.array_size,))

            mlp_model_exp = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'swish')
            output_exp = mlp_model_exp.GRU_model('swish')
            model_exp = Model(inputs=input_rnn, outputs=output_exp,  name='swish_loss')
            output1 = model_exp(input_rnn)

            mlp_model_line = GRU_model(input_rnn, self.rnn_layers, self.time_steps,'linear')
            output_line = mlp_model_line.GRU_model('linear')
            model_line = Model(inputs=input_rnn, outputs=output_line, name='linear_loss')
            output2 = model_line(input_rnn)

            mlp_model_sine = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'sine')
            output_sine = mlp_model_sine.GRU_model('sine')
            model_sine = Model(inputs=input_rnn, outputs=output_sine, name='sine_loss')
            output3 = model_sine(input_rnn)

            mlp_model_cos = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'cos')
            output_cos = mlp_model_cos.GRU_model('cos')
            model_cos = Model(inputs=input_rnn, outputs=output_cos, name='cos_loss')
            output4 = model_cos(input_rnn)

            mlp_model_tanh = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'tanh')
            output_tanh = mlp_model_tanh.GRU_model('tanh')
            model_tanh = Model(inputs=input_rnn, outputs=output_tanh, name='tanh_loss')
            output5 = model_tanh(input_rnn)

            mlp_model_relu = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'relu')
            output_relu = mlp_model_relu.GRU_model('relu')
            model_relu = Model(inputs=input_rnn, outputs=output_relu, name='relu_loss')
            output6 = model_relu(input_rnn)

            mlp_model_sigmoid = GRU_model(input_rnn, self.rnn_layers, self.time_steps, 'sigmoid')
            output_sigmoid = mlp_model_sigmoid.GRU_model('sigmoid')
            model_sigmoid = Model(inputs=input_rnn, outputs=output_sigmoid, name='sigmoid_loss')
            output7 = model_sigmoid(input_rnn)

            final_input = Concatenate()([output1, output2, output3, output4,
                                         output5, output6, output7])

            soft_output = Dense(7, activation='softmax')(final_input)

            final_input = Multiply()([final_input, soft_output])

            final_output = Dense(1, activation='linear')(final_input)

            proposed_model = Model(inputs=input_rnn, outputs=final_output, name='Prediction_loss')

            print(proposed_model.summary())

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model


class mlp_models():

    def __init__(self, trainingX, valX, trainingY, valY,
                 mlp_layers, time_steps, batch_size,
                 early_stopping, model_path, method, var, epoch, lr, reg_lambda= 0.00001):

        trainingX, valX = np.reshape(trainingX, (trainingX.shape[0], trainingX.shape[1]*trainingX.shape[2])), \
                              np.reshape(valX, (valX.shape[0], valX.shape[1]*valX.shape[2]))


        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.mlp_layers = mlp_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[1]
        self.batch_size = batch_size
        self.epoch = epoch
        self.var = var
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.method = method

    def proposed_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s' %(self.method, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s' %(self.method, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_mlp = Input(shape=(self.array_size,),)

            mlp_model_exp = MLP_model(input_mlp, self.mlp_layers, 'swish')
            output_exp = mlp_model_exp.model('swish')
            model_exp = Model(inputs=input_mlp, outputs=output_exp, name='swish_loss')
            output1=model_exp(input_mlp)

            mlp_model_line = MLP_model(input_mlp, self.mlp_layers, 'linear')
            output_line = mlp_model_line.model('linear')
            model_line = Model(inputs=input_mlp, outputs=output_line, name='linear_loss')
            output2 = model_line(input_mlp)

            mlp_model_sine = MLP_model(input_mlp, self.mlp_layers,  'sine')
            output_sine = mlp_model_sine.model('sine')
            model_sine = Model(inputs=input_mlp, outputs=output_sine, name='sine_loss')
            output3 = model_sine(input_mlp)

            mlp_model_cos = MLP_model(input_mlp, self.mlp_layers, 'cos')
            output_cos = mlp_model_cos.model('cos')
            model_cos = Model(inputs=input_mlp, outputs=output_cos, name='cos_loss')
            output4 = model_cos(input_mlp)

            mlp_model_tanh = MLP_model(input_mlp, self.mlp_layers, 'tanh')
            output_tanh = mlp_model_tanh.model('tanh')
            model_tanh = Model(inputs=input_mlp, outputs=output_tanh, name='tanh_loss')
            output5 = model_tanh(input_mlp)

            mlp_model_relu = MLP_model(input_mlp, self.mlp_layers, 'relu')
            output_relu = mlp_model_relu.model('relu')
            model_relu = Model(inputs=input_mlp, outputs=output_relu, name='relu_loss')
            output6 = model_relu(input_mlp)

            mlp_model_sigmoid = MLP_model(input_mlp, self.mlp_layers, 'sigmoid')
            output_sigmoid = mlp_model_sigmoid.model('sigmoid')
            model_sigmoid = Model(inputs=input_mlp, outputs=output_sigmoid, name='sigmoid_loss')
            output7 = model_sigmoid(input_mlp)

            final_input = Concatenate()([output1, output2, output3, output4,
                                       output5, output6, output7])

            soft_output = Dense(7, activation='softmax')(final_input)

            final_input = Multiply()([final_input, soft_output])

            final_output = Dense(1, activation='linear')(final_input)

            proposed_model = Model(inputs = input_mlp, outputs = final_output, name='Prediction_loss')

            print(proposed_model.summary())
            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX,  self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model


class simple_rnn_models():

    def __init__(self, trainingX,  valX, trainingY, valY,
                 rnn_layers, time_steps, batch_size, activation, output_activation,
                 early_stopping, model_path, method, var, epoch, lr, reg_lambda= 0.00001):

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.rnn_layers = rnn_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[2]
        self.batch_size = batch_size
        self.activation=activation
        self.output_activation = output_activation
        self.epoch = epoch
        self.var = var
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.method = method
        self.reg_lambda = reg_lambda

    def rnn_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            %(self.method, self.method, self.activation, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                                     %(self.method, self.method, self.activation, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            inputs = Input(shape=(self.time_steps, self.array_size,))

            rnn_model = RNN_model(inputs, self.rnn_layers, self.time_steps, self.activation)
            output = rnn_model.RNN_model(self.activation)

            output = Dense(4, activation=self.activation)(output)
            final_output = Dense(1, activation=self.output_activation)(output)

            proposed_model = Model(inputs = inputs, outputs = final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

    def lstm_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            %(self.method, self.method, self.activation, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                                     %(self.method, self.method, self.activation, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            inputs = Input(shape=(self.time_steps, self.array_size,))

            lstm_model = LSTM_model(inputs, self.rnn_layers, self.time_steps, self.activation)
            output = lstm_model.LSTM_model(self.activation)

            output = Dense(4, activation=self.activation)(output)
            final_output = Dense(1, activation=self.output_activation)(output)

            proposed_model = Model(inputs = inputs, outputs = final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')


        return final_model

    def gru_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            % (self.method, self.method, self.activation, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                                     % (self.method, self.method, self.activation, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                            batch_size=self.batch_size, verbose=2,
                            shuffle=False, validation_data=(self.valX, self.valY),
                            callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            inputs = Input(shape=(self.time_steps, self.array_size,))

            gru_model = GRU_model(inputs, self.rnn_layers, self.time_steps, self.activation)
            output = gru_model.GRU_model(self.activation)

            output = Dense(4, activation=self.activation)(output)
            final_output = Dense(1, activation=self.output_activation)(output)

            proposed_model = Model(inputs=inputs, outputs=final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model

class simple_MLP_model():

    def __init__(self, trainingX, valX, trainingY, valY, activation,
                 output_activation, mlp_layers, time_steps, batch_size,
                 early_stopping, model_path, method, var, epoch, lr, reg_lambda=0.00001):

        trainingX, valX = np.reshape(trainingX, (trainingX.shape[0], trainingX.shape[1] * trainingX.shape[2])), \
                          np.reshape(valX, (valX.shape[0], valX.shape[1] * valX.shape[2]))

        self.trainingX = trainingX
        self.valX = valX
        self.trainingY = trainingY
        self.valY = valY
        self.mlp_layers = mlp_layers
        self.time_steps = time_steps
        self.array_size = trainingX.shape[1]
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.epoch = epoch
        self.var = var
        self.early_stopping = early_stopping
        self.model_path = model_path
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.method = method

    def mlp_model(self):

        if os.path.isfile(self.model_path + '.h5'):

            final_model = load_model(self.model_path + '.h5')

        elif os.path.isfile('model/%s/model_%s_%s_%s_%s'
                            %(self.method, self.method, self.activation, self.var, 1) + '.h5'):

            final_model = load_model('model/%s/model_%s_%s_%s_%s'
                                     %(self.method, self.method, self.activation, self.var, 1) + '.h5')

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            final_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            final_model.fit(self.trainingX, self.trainingY, epochs=300,
                               batch_size=self.batch_size, verbose=2,
                               shuffle=False, validation_data=(self.valX, self.valY),
                               callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        else:

            input_mlp = Input(shape=(self.array_size,), )

            mlp_model = MLP_model(input_mlp, self.mlp_layers, self.activation)
            output = mlp_model.model(self.activation)

            output = Dense(4, activation=self.activation)(output)
            final_output = Dense(1, activation=self.output_activation)(output)

            proposed_model = Model(inputs = input_mlp, outputs = final_output)

            optimizer = adam(lr=self.lr, epsilon=None, decay=0.0, amsgrad=False)

            proposed_model.compile(loss='mean_squared_error', optimizer=optimizer)

            early_stop = [EarlyStopping(monitor='val_loss', min_delta=0,
                                        patience=self.early_stopping, verbose=1, mode='auto'),
                          ModelCheckpoint(self.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
                          ]

            proposed_model.fit(self.trainingX, self.trainingY, epochs=self.epoch, batch_size=self.batch_size, verbose=2,
                           shuffle=False, validation_data=(self.valX, self.valY), callbacks=early_stop)

            final_model = load_model(self.model_path + '.h5')

        return final_model


class data_preparing():

    def __init__(self, X, Y, testX, testY, time_steps, ratio=0.8):

        self.X = X
        self.Y = Y
        self.testX = testX
        self.testY = testY
        self.time_steps = time_steps
        self.array_size = 1
        self.ratio = ratio

    def data_splitting(self, dataX, dataY):

        #### data partitioning
        train_len = int(np.round(len(dataY) * self.ratio, 0))
        trainX, valX = dataX[:train_len], dataX[train_len:]
        trainY, valY = dataY[:train_len], dataY[train_len:]

        return trainX, valX,\
               trainY, valY

    def prepare_timeseries(self, X, Y):
        dataX, dataY = [], []
        length = len(Y) - self.time_steps
        for i in range(0, length):
            dataX.append(X[i:(i + self.time_steps)])
            dataY.append(Y[(i + self.time_steps)])

        return np.array(dataX), np.array(dataY)

    def data_preprocessing(self):

        dataX, dataY = self.prepare_timeseries(self.X, self.Y)
        testX, testY = self.prepare_timeseries(self.testX, self.testY)

        trainX, valX, trainY, valY = self.data_splitting(dataX, dataY)
        trainX, valX, testX = np.reshape(trainX, (trainX.shape[0], self.time_steps, self.array_size)), \
                              np.reshape(valX, (valX.shape[0], self.time_steps, self.array_size)),\
                              np.reshape(testX, (testX.shape[0], self.time_steps, self.array_size))

        return trainX, valX, trainY, valY, testX, testY
