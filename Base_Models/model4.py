### model4: 2layered LSTM(takes 1input)

import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Dropout,BatchNormalization
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D

class Layered_LSTM:


    def __init__(self,n_timesteps,n_features,n_outputs):
        self.n_timesteps=n_timesteps
        self.n_features=n_features
        self.n_outputs=n_outputs
        input_lstm=Input(shape=(n_timesteps,n_features))
        lstm_1=LSTM(128,activation = 'tanh',return_sequences = True)(input_lstm)
        drop_1=Dropout(0.2)(lstm_1)
        batch_1=BatchNormalization()(drop_1)

        lstm_2=LSTM(128,input_shape=(n_timesteps,n_features),activation='tanh')(batch_1)
        drop_2=Dropout(0.2)(lstm_2)
        batch_2=BatchNormalization()(drop_2)

        dense=Dense(32,activation='relu')(batch_2)
        drop_3=Dropout(0.2)(dense)
        
        output_lstm=Dense(n_outputs,activation = 'softmax')(drop_3)
        self.lstm_model=Model(input_lstm,output_lstm)

            

    def do_compile(self,trainX,testX,trainy_one_hot,testy_one_hot):
        opt =keras.optimizers.Adam(learning_rate=1e-3,decay=1e-5)
        self.lstm_model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics = ['accuracy'])
        model_history=self.lstm_model.fit(trainX, trainy_one_hot, epochs=30, batch_size=64,validation_data=(testX,testy_one_hot))
        return self.lstm_model

    def prediction(self,testX):
        predy=self.lstm_model.predict(testX)
        predy=np.argmax(predy, axis=-1)
        return predy