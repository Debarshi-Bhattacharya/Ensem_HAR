### model3: Timdistributed ConvLSTM(takes 1input)


import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters



class Conv_LSTM:


        def __init__(self,n_timesteps,n_features,n_outputs):
            self.n_timesteps=n_timesteps
            self.n_features=n_features
            self.n_outputs=n_outputs
            self.n_steps, self.n_length = 4, 32
        
        
        def build_model(self,hp):  
            model =Sequential([
            TimeDistributed(Conv1D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values = [3,5,7,9]),
                activation='relu',
                input_shape=(None,self.n_length,self.n_features)
            )),
            TimeDistributed(Conv1D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
                kernel_size=hp.Choice('conv_2_kernel', values = [3,5,7]),
                activation='relu'
            )),
            TimeDistributed(Conv1D(
                filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=16),
                kernel_size=hp.Choice('conv_3_kernel', values = [3,5,7]),
                activation='relu'
            )),
            TimeDistributed(Dropout(0.5)),
            TimeDistributed(MaxPooling1D(pool_size=2)),
            TimeDistributed(Flatten()),
            LSTM(units=hp.Int('hidden_units',min_value=100,max_value=300,step=20), return_sequences=True),
            Dropout(0.5,seed=0),
            LSTM(units=hp.Int('hidden_units',min_value=100,max_value=300,step=20)),
            Dropout(0.5, seed=1),
            Dense(
                units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
                activation='relu'
            ),
            Dense(self.n_outputs,activation='softmax')
        ])
            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),loss='categorical_crossentropy',metrics=['accuracy'])

            return model
            # self.model = Sequential()
            # self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=5, activation='relu'), input_shape=(None,self.n_length,self.n_features)))
            # self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu')))
            # self.model.add(TimeDistributed(Conv1D(filters=48, kernel_size=7, activation='relu')))
            # self.model.add(TimeDistributed(Dropout(0.5)))
            # self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            # self.model.add(TimeDistributed(Flatten()))
            # self.model.add(LSTM(280,return_sequences=True))
            # self.model.add(Dropout(0.5))
            # self.model.add(LSTM(280))
            # self.model.add(Dropout(0.5))
            # self.model.add(Dense(96, activation='relu'))
            # self.model.add(Dense(n_outputs, activation='softmax'))
            
            
        def do_compile(self,trainX,testX,trainy_one_hot,testy_one_hot):
            trainX= trainX.reshape((trainX.shape[0], self.n_steps, self.n_length,self.n_features))
            testX = testX.reshape((testX.shape[0], self.n_steps, self.n_length, self.n_features))
            tuner_search=RandomSearch(self.build_model,objective='val_accuracy',max_trials=10,directory='output',project_name="HAR_ConvLstm")
            tuner_search.search(trainX,trainy_one_hot,epochs=10,batch_size=32,validation_data= (testX,testy_one_hot))
            # opt = keras.optimizers.Adam(learning_rate=1e-2)
            # self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            self.best_model=tuner_search.get_best_models(num_models=1)[0]
            self.best_model.fit(trainX,trainy_one_hot,epochs=30,validation_data= (testX,testy_one_hot), initial_epoch=10)
            return self.best_model
        
        def prediction(self,testX):
            testX = testX.reshape((testX.shape[0], self.n_steps, self.n_length, self.n_features))
            predy=self.best_model.predict(testX)
            predy=np.argmax(predy, axis=-1)
            return predy