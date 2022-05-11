### model1: 3levels on CNN (4layers in each)(takes 4input) 

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
class cnn:

    def __init__(self,n_timesteps,n_features,n_outputs,weights=None):
        self.n_timesteps=n_timesteps
        self.n_features=n_features
        self.n_outputs=n_outputs
        self.class_weight=weights
        ##Level_1
        # layer 1
        inputs1_1= Input(shape=(n_timesteps,n_features))##
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


        ##Level_3
        # layer 1
        conv3_1 = Conv1D(filters=32, kernel_size=3, activation='relu')(pool_2)
        # layer 2
        conv3_2 = Conv1D(filters=32, kernel_size=5, activation='relu')(pool_2)
        # layer 3
        conv3_3 = Conv1D(filters=32, kernel_size=7, activation='relu')(pool_2)
        # layer 4

        conv3_4 = Conv1D(filters=32, kernel_size=9, activation='relu')(pool_2) 
        # merge2
        merged_3 = concatenate([conv3_1,conv3_2,conv3_3,conv3_4],axis=1)

        #maxpool2
        pool_3=MaxPooling1D(pool_size=5)(merged_3)


        #flatten
        flat_cnn=Flatten()(pool_3)
        
        ##dense layer
        dense = Dense(512, activation='relu')(flat_cnn)
        outputs = Dense(n_outputs, activation='softmax')(dense)
        
        ##MODEL
        self.cnn3_model = Model([inputs1_1, inputs1_2, inputs1_3,inputs1_4], outputs)
        
    def do_compile(self,trainX,testX,trainy_one_hot,testy_one_hot,batch=64):
        self.cnn3_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        self.model_history=self.cnn3_model.fit(x=[trainX,trainX,trainX,trainX], y=trainy_one_hot, epochs=30, batch_size=batch,class_weight=self.class_weight,validation_data= ([testX,testX,testX,testX],testy_one_hot))
        self.cnn3_model.summary()
        return self.cnn3_model
    
    def prediction(self,testX):
        predy=self.cnn3_model.predict([testX,testX,testX,testX])
        predy=np.argmax(predy, axis=-1)
        return predy
        