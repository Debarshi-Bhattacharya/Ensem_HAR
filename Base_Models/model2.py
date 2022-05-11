### model2: 3levels on CNN (4layers in each) , parallaly 1LSTM layer (takes 5inputs)

import numpy as np
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
class cnn_lstm:
    
    
    def __init__(self,n_timesteps,n_features,n_outputs,weights=None):
        self.n_timesteps=n_timesteps
        self.n_features=n_features
        self.n_outputs=n_outputs
        self.class_weight=weights
    ###########################___CNN___######################################

        ##Level_1
        # layer 1
        inputs1_1= Input(shape=(n_timesteps,n_features))##128,9
        conv1_1 = Conv1D(filters=128, kernel_size=3, activation='relu')(inputs1_1) ##none,126,128
            # layer 2
        inputs1_2= Input(shape=(n_timesteps,n_features))
        conv1_2 = Conv1D(filters=128, kernel_size=5, activation='relu')(inputs1_2)##124,128
            # layer 3
        inputs1_3= Input(shape=(n_timesteps,n_features))
        conv1_3 = Conv1D(filters=128, kernel_size=7, activation='relu')(inputs1_3)##122,128
            # layer 4
        inputs1_4= Input(shape=(n_timesteps,n_features))
        conv1_4 = Conv1D(filters=128, kernel_size=9, activation='relu')(inputs1_4)##120,128
            
            # merge1
        merged_1 = concatenate([conv1_1,conv1_2,conv1_3,conv1_4],axis=1)
            
            #maxpool1
        pool_1=MaxPooling1D(pool_size=5)(merged_1)

        ##Level_2
        # layer 1
        conv2_1 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool_1)
        # layer 2
        conv2_2 = Conv1D(filters=64, kernel_size=5, activation='relu')(pool_1)
        # layer 3
        conv2_3 = Conv1D(filters=64, kernel_size=7, activation='relu')(pool_1)
        # layer 4

        conv2_4 = Conv1D(filters=64, kernel_size=9, activation='relu')(pool_1) 
        # merge2
        merged_2 = concatenate([conv2_1,conv2_2,conv2_3,conv2_4],axis=1)
            
        #maxpool2
        pool_2=MaxPooling1D(pool_size=5)(merged_2)
            
        #flatten
        flat_cnn=Flatten()(pool_2)

        dense1 = Dense(512, activation='relu')(flat_cnn)
        cnn_drop=Dropout(0.5)(dense1)

    #############################___LSTM___###################################################
        inputs_LSTM=Input(shape=(n_timesteps,n_features))
        LSTM_model=LSTM(100)(inputs_LSTM)
        flat_lstm=Flatten()(LSTM_model)
        dense2=Dense(512, activation='relu')(flat_lstm)
        lstm_drop=Dropout(0.5)(dense2)



    ############################___CNN+LSTM___#################################################
        cnn_lstm=concatenate([cnn_drop,lstm_drop])
        flat_cnnlstm=Flatten()(cnn_lstm)
        cnnlstm_drop=Dropout(0.5)(flat_cnnlstm)
        outputs = Dense(n_outputs, activation='softmax')(cnnlstm_drop)


        self.cnnlstm_model = Model(inputs=[inputs1_1, inputs1_2, inputs1_3,inputs1_4,inputs_LSTM], outputs=outputs)


    def do_compile(self,trainX,testX,trainy_one_hot,testy_one_hot):
        self.cnnlstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model_history=self.cnnlstm_model.fit(x=[trainX,trainX,trainX,trainX,trainX], y=trainy_one_hot, epochs=30, batch_size=64,class_weight=self.class_weight,validation_data= ([testX,testX,testX,testX,testX],testy_one_hot))
        return self.cnnlstm_model

    def prediction(self,testX):
        predy=self.cnnlstm_model.predict([testX,testX,testX,testX,testX])
        predy=np.argmax(predy, axis=-1)
        return predy