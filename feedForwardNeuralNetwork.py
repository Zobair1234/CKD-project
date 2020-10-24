import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('final.csv')

x = data.drop('Class', axis = 1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
for i in range (50):
	model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=500)

model.evaluate(x_test, y_test, verbose=0)

pred = model.predict(x_test)

print(pred)



