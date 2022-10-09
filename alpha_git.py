# -*- coding: utf-8 -*-


"""import numpy as np
import matplotlib as mp"""
import pandas as pd

import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from keras.models import Sequential
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

from matplotlib import pyplot
from tensorflow.keras import callbacks

df_svt = pd.read_csv("D:/me/Sicherung/Finanzen/Alpha/Daten/svt.csv", sep=";")




df_svt = df_svt.iloc[6980:]
df_svt = df_svt.iloc[:8158]

df_y = pd.DataFrame(df_svt.y)


df_y_train = df_y.iloc[:6501]
df_y_test = df_y.iloc[6501:8158]

y_train = df_y_train.to_numpy()
y_test = df_y_test.to_numpy()


df_svt = df_svt.drop(columns=["y"])
df_svt = df_svt.drop(columns=["Date"])

df_x_train = df_svt.iloc[:6501]
df_x_test = df_svt.iloc[6501:8158]

x_train = df_x_train.to_numpy()
x_test = df_x_test.to_numpy()


train_X = x_train.reshape(6501,1,7)
test_X = x_test.reshape(1657,1,7)

train_Y = y_train.reshape(6501,1)
test_Y = y_test.reshape(1657,1)




trainX = []
trainY = []
testX = []
testY = []



ts = 20
timestep = ts

def create_dataset(datasetX,datasetY, timestep=10):
    dataX, dataY = [],[]
    
    for i in range(len(datasetX)-timestep-1):
        a = datasetX[i:(i+timestep),0]
        dataX.append(a)
        dataY.append(datasetY[i+timestep-1,0])
    return np.array(dataX), np.array(dataY)

timestep=ts
trainX, trainY = create_dataset(train_X, train_Y, timestep)
timestep=ts
testX, testY = create_dataset(test_X,test_Y,timestep)


#print(trainX.shape[2])


#print(testX[0])

    

print("X ", trainX[1])

print("Y ", trainY[1])

"""
i=0
for i in range (0,10):
        print("Y ",trainY[i])

"""




model = tf.keras.Sequential()

model.add(Dense(12, input_shape=(7,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


"""
model.add(LSTM(units=70,recurrent_activation="sigmoid", return_sequences=True , input_shape=(train_X.shape[1],train_X.shape[2]) ))



#model.add(Dropout(0.2))
model.add(LSTM(units=70, recurrent_activation="sigmoid",return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=30, recurrent_activation="sigmoid", return_sequences=False,activation="softmax" ))
#model.add(Dropout(0.2))"""


#model.add(Dense(units=1))


#model.compile(optimizer="adam",loss="mae")
model.compile(optimizer="adam",loss="mean_squared_error", metrics="accuracy")
#model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics="accuracy")


model.fit(x_train, y_train, epochs=20, verbose=1, batch_size =10)
model.summary()
accuracy= model.evaluate(x_train,y_train)
#print('Accuracy: %.2f' % (accuracy*100))





train = model.predict(x_train, batch_size = 10)
#predict = model.predict(test_X, batch_size = 10)

print(train)
#print(predict)
i=0
for i in range(25):
    print("train", i,": ", train[i], "train_Y ",i,": ",train_Y[i])
    
    
 

#print(predict)
#print(testX.shape)
#print(testY.shape)
#print(predict.shape)




    
